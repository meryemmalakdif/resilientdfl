from .base import BaseTrigger
import torch

class DBATrigger(BaseTrigger):
    """
    Implements a dynamic distributed trigger where clients are assigned a shard
    in a round-robin fashion using the modulo operator.
    """
    def __init__(self, client_id, shard_locations, global_position=(26, 26), 
                 patch_size=(2, 2), color=(1.0, 0.0, 0.0)):
        """
        Initializes the DBA trigger using a dynamic assignment rule.

        Args:
            client_id (int): The ID of the current malicious client.
            shard_locations (list[tuple]): A list of the *unique* (x_offset, y_offset)
                                           tuples for the available shards.
            global_position (tuple): Top-left (x, y) of the entire trigger area.
            patch_size (tuple): The (width, height) of a single client's patch.
            color (tuple): The (R, G, B) color of the trigger, in [0.0, 1.0].
        """
        self.client_id = client_id
        num_shards = len(shard_locations)

        if num_shards == 0:
            raise ValueError("The shard_locations list cannot be empty.")

        # Dynamic assignment using the modulo operator
        shard_index = self.client_id % num_shards
        
        # Get the relative offset for this client's assigned shard
        x_offset, y_offset = shard_locations[shard_index]

        # Calculate the absolute on-image position for this client's patch
        local_position = (global_position[0] + x_offset, global_position[1] + y_offset)
        
        # The pattern is just the color, reshaped for broadcasting
        local_pattern = torch.tensor(color).view(-1, 1, 1)

        super().__init__(local_position, patch_size, local_pattern, alpha=1.0)

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies this client's specific portion of the distributed trigger.
        (This method is identical to the previous implementation).

        Args:
            image (torch.Tensor): A clean image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: The image with the trigger segment embedded.
        """
        poisoned_image = image.clone()
        _, img_h, img_w = poisoned_image.shape
        patch_w, patch_h = self.size
        x_pos, y_pos = self.position

        pattern = self.pattern.to(image.device)
        
        y_start = min(y_pos, img_h)
        y_end = min(y_pos + patch_h, img_h)
        x_start = min(x_pos, img_w)
        x_end = min(x_pos + patch_w, img_w)
        
        region = poisoned_image[:, y_start:y_end, x_start:x_end]
        
        poisoned_image[:, y_start:y_end, x_start:x_end] = \
            self.alpha * pattern + (1.0 - self.alpha) * region
            
        return poisoned_image