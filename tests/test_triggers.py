import torch
from src.attacks.triggers.patch import PatchTrigger
from src.attacks.triggers.distributed import DBATrigger

import pytest

def test_patch_trigger():
    """
    Test the PatchTrigger by applying it to a sample image tensor.
    """
    # Create a sample black image tensor (3 channels, 32x32 pixels)
    # Models typically expect float32 tensors.
    clean_tensor = torch.zeros((3, 32, 32), dtype=torch.float32)

    # Initialize the trigger (using default red patch at bottom-right)
    patch_trigger = PatchTrigger(position=(28, 28), size=(3, 3))

    # Apply the trigger
    poisoned_tensor = patch_trigger.apply(clean_tensor)

    # Check that the poisoned tensor has the same shape and dtype as the clean tensor
    assert poisoned_tensor.shape == clean_tensor.shape, "Shape mismatch after applying trigger"
    assert poisoned_tensor.dtype == clean_tensor.dtype, "Dtype mismatch after applying trigger"

    # check that the patch area has been modified
    patch_area = poisoned_tensor[:, 28:31, 28:31]
    assert torch.allclose(patch_area, torch.tensor([[[1.0]], [[0.0]], [[0.0]]])), "Patch area not correctly modified"


def test_dba_trigger():
    """
    Test the DBATrigger by applying it to a sample image tensor.
    """
    # Create a sample black image tensor (3 channels, 32x32 pixels)
    clean_tensor = torch.zeros((3, 32, 32), dtype=torch.float32)

    # Define shard locations (relative offsets)
    shard_locations = [(0, 0), (5, 0), (0, 5), (5, 5)]

    # Initialize the trigger for client_id=2
    dba_trigger = DBATrigger(client_id=2, shard_locations=shard_locations,
                             global_position=(26, 26), patch_size=(2, 2),
                             color=(0.0, 1.0, 0.0))  # green patch

    # Apply the trigger
    poisoned_tensor = dba_trigger.apply(clean_tensor)

    # Check that the poisoned tensor has the same shape and dtype as the clean tensor
    assert poisoned_tensor.shape == clean_tensor.shape, "Shape mismatch after applying trigger"
    assert poisoned_tensor.dtype == clean_tensor.dtype, "Dtype mismatch after applying trigger"

    # check that the correct shard area has been modified
    # For client_id=2 and given shard_locations, the offset is (0,5)
    expected_x_start = 26 + 0
    expected_y_start = 26 + 5
    expected_x_end = expected_x_start + 2
    expected_y_end = expected_y_start + 2

    patch_area = poisoned_tensor[:, expected_y_start:expected_y_end, expected_x_start:expected_x_end]
    assert torch.allclose(patch_area, torch.tensor([[[0.0]], [[1.0]], [[0.0]]])), "Shard area not correctly modified"