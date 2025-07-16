import cdtools
import cdtools.tools.distributed as dist
import pytest
import os

"""
This file contains several tests that are relevant to running
multi-GPU operations in CDTools. 



"""

@pytest.mark.multigpu
def test_reconstruction_quality(lab_ptycho_cxi,
                                multigpu_script_1,
                                tmp_path,
                                show_plot):
    """
    Run a multi-GPU test based on fancy_ptycho_speed_test.py and make 
    sure the final reconstructed loss using 2 GPUs is similar to 1 GPU.

    This test requires us to have 2 NVIDIA GPUs and makes use of the
    multi-GPU speed test.

    If this test fails, it indicates that the reconstruction quality
    is getting noticably worse with increased GPU counts. This may 
    be a symptom of a synchronization/broadcasting issue between the 
    different GPUs.
    """
    # Pass the cxi directory to the reconstruction script
    os.environ['CDTOOLS_TESTING_DATA_PATH'] = lab_ptycho_cxi

    # Set up and run a distributed speed test
    world_sizes = [1, 2]
    runs = 5
    file_prefix = 'speed_test'

    # Define a temporary directory
    temp_dir = str(tmp_path / "reconstruction")

    results = dist.run_speed_test(world_sizes=world_sizes,
                                  runs=runs,
                                  script_path=multigpu_script_1,
                                  output_dir=temp_dir,
                                  file_prefix=file_prefix,
                                  show_plot=show_plot,
                                  delete_output_files=True)

    # Ensure that both single and 2 GPU results produce losses lower than
    # a threshold value of 0.0013. This is the same threshold used in
    # test_fancy_ptycho.py
    loss_mean = results[0]
    assert loss_mean[0] < 0.0013
    assert loss_mean[1] < 0.0013

    # Check if the two losses are similar to each other by seeing if their
    # mean +- standard deviation intervals overlap with each other
    loss_std = results[1]
    single_gpu_loss_min = loss_mean[0] - loss_std[0]
    single_gpu_loss_max = loss_mean[0] + loss_std[0]
    multi_gpu_loss_min = loss_mean[1] - loss_std[1]
    multi_gpu_loss_max = loss_mean[1] + loss_std[1]
    has_overlap_loss = min(single_gpu_loss_max, multi_gpu_loss_max)\
                       > max(single_gpu_loss_min, multi_gpu_loss_min)

    print(f'Single GPU final loss: {loss_mean[0]} +- {loss_std[0]}')
    print(f'Two GPU final loss: {loss_mean[1]} +- {loss_std[1]}')
    print(f'Overlap between the mean +- std of the single/multi GPU losses: {has_overlap_loss}')

    assert has_overlap_loss

    # Also make sure that we actually get some kind of speed up with 
    # multiple GPUs...
    speed_mean = results[2]
    speed_std = results[3]

    single_gpu_speed_min = speed_mean[0] - speed_std[0]
    single_gpu_speed_max = speed_mean[0] + speed_std[0]
    multi_gpu_speed_min = speed_mean[1] - speed_std[1]
    multi_gpu_speed_max = speed_mean[1] + speed_std[1]
    has_overlap_speed = min(single_gpu_speed_max, multi_gpu_speed_max)\
                        > max(single_gpu_speed_min, multi_gpu_speed_min)

    print(f'Single GPU runtime: {speed_mean[0]} +- {speed_std[0]}')
    print(f'Two GPU runtime: {speed_mean[1]} +- {speed_std[1]}')
    print(f'Overlap between the mean +- std of the single/multi GPU runtimes: {has_overlap_speed}')

    assert speed_mean[0] < speed_mean[1]
    assert not has_overlap_speed

    # Clear the environment variable we created here
    os.environ.pop('CDTOOLS_TESTING_DATA_PATH')


