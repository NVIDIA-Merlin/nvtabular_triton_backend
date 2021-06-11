import contextlib
import signal
import subprocess
import time
from distutils.spawn import find_executable

import cudf
import numpy as np
import pytest
import tritonclient
import tritonclient.grpc as grpcclient
from cudf.tests.utils import assert_eq

import nvtabular as nvt
import nvtabular.inference.triton as nvt_triton
import nvtabular.ops as ops

_TRITON_SERVER_PATH = find_executable("tritonserver")


@contextlib.contextmanager
def run_triton_server(modelpath):
    cmdline = [_TRITON_SERVER_PATH, "--model-repository", modelpath]
    with subprocess.Popen(cmdline) as process:
        try:
            with grpcclient.InferenceServerClient("localhost:8001") as client:
                # wait until server is ready
                for _ in range(60):
                    if process.poll() is not None:
                        retcode = process.returncode
                        raise RuntimeError(f"Tritonserver failed to start (ret={retcode})")

                    try:
                        ready = client.is_server_ready()
                    except tritonclient.utils.InferenceServerException:
                        ready = False

                    if ready:
                        yield client
                        return

                    time.sleep(1)

                raise RuntimeError("Timed out waiting for tritonserver to become ready")
        finally:
            # signal triton to shutdown
            process.send_signal(signal.SIGINT)


def test_error_handling(tmpdir):
    df = cudf.DataFrame({"x": np.arange(10), "y": np.arange(10)})

    def custom_transform(col):
        if len(col) == 2:
            raise ValueError("Lets cause some problems")
        return col

    features = ["x", "y"] >> ops.FillMissing() >> ops.Normalize() >> custom_transform
    workflow = nvt.Workflow(features)
    workflow.fit(nvt.Dataset(df))

    model_name = "test_error_handling"
    nvt_triton.generate_nvtabular_model(workflow, model_name, tmpdir + f"/{model_name}", backend="nvtabular")

    with run_triton_server(tmpdir) as client:
        inputs = nvt_triton.convert_df_to_triton_input(["x", "y"], df[:2])
        with pytest.raises(tritonclient.utils.InferenceServerException) as exception_info:
            client.infer(model_name, inputs)

        assert "ValueError: Lets cause some problems" in str(exception_info.value)


def test_tritonserver_inference_string(tmpdir):
    df = cudf.DataFrame({"user": ["aaaa", "bbbb", "cccc", "aaaa", "bbbb", "aaaa"]})
    features = ["user"] >> ops.Categorify()
    workflow = nvt.Workflow(features)

    # fit the workflow and test on the input
    dataset = nvt.Dataset(df)
    workflow.fit(dataset)

    local_df = workflow.transform(dataset).to_ddf().compute(scheduler="synchronous")
    model_name = "test_inference_string"
    nvt_triton.generate_nvtabular_model(workflow, model_name, tmpdir + f"/{model_name}", backend="nvtabular")

    inputs = nvt_triton.convert_df_to_triton_input(["user"], df)
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs)
        user_features = response.as_numpy("user")
        triton_df = cudf.DataFrame({"user": user_features.reshape(user_features.shape[0])})
        assert_eq(triton_df, local_df)
