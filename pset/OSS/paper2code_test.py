import unittest
import torch
import numpy as np
from model import search_OSS as search_OSS_generated
from model_ref import search_OSS as search_OSS_original
from model import get_teacher_trajectory, dynamic_programming_oss, student_teacher_approximation, retrieve_optimal_trajectory
from model_ref import get_teacher_trajectory as get_teacher_trajectory_ref
from model_ref import dynamic_programming_oss as dynamic_programming_oss_ref
from model_ref import student_teacher_approximation as student_teacher_approximation_ref
from model_ref import retrieve_optimal_trajectory as retrieve_optimal_trajectory_ref


class TestOSSEquivalence(unittest.TestCase):
    """Test suite to verify the equivalence of search_OSS implementations."""

    def setUp(self):
        """Set up common test fixtures."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test inputs
        self.batch_size = 2
        self.channels = 3
        self.height = 4
        self.width = 4
        self.teacher_steps = 10
        self.student_steps = 5

        # Create random input tensor
        self.z = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        # Create random class embeddings
        self.class_emb = torch.randn(self.batch_size, 12)  # Assuming 512-dim embeddings
        
        # Create a mock model that returns random velocities
        class MockModel:
            self.teacher_steps = 10
            def __init__(self):
                self.fm_steps = torch.linspace(0, 1, 10 + 1)
            
            @torch.no_grad()
            def __call__(self, z, t, class_emb, model_kwargs):
                return z*3+4

        self.model = MockModel()
        self.device = torch.device('cpu')
        self.model_kwargs = {}

    def test_teacher_trajectory_equivalence(self):
        """Test that both implementations produce the same teacher trajectory."""
        # Call both implementations
        traj_generated = get_teacher_trajectory(
            self.model,
            self.z,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.model_kwargs
        )
        
        traj_original = get_teacher_trajectory_ref(
            self.model,
            self.z,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.model_kwargs
        )

        # Check shapes
        self.assertEqual(traj_generated.shape, traj_original.shape,
                         "Teacher trajectory shapes differ")
        
        # Check values
        self.assertTrue(torch.allclose(traj_generated, traj_original),
                        "Teacher trajectories differ")

    def test_dynamic_programming_equivalence(self):
        """Test that both implementations produce the same dynamic programming results."""
        # First get teacher trajectory
        traj_tea = get_teacher_trajectory(
            self.model,
            self.z,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.model_kwargs
        )

        # Call both implementations
        steps_generated = dynamic_programming_oss(
            self.model,
            self.z,
            traj_tea,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.student_steps,
            self.model_kwargs
        )
        
        steps_original = dynamic_programming_oss_ref(
            self.model,
            self.z,
            traj_tea,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.student_steps,
            self.model_kwargs
        )

        # Check that both implementations return the same number of steps
        self.assertEqual(len(steps_generated), len(steps_original),
                         "Number of batches in output differs")
        
        # Check each batch's steps
        for batch_idx in range(len(steps_generated)):
            self.assertEqual(len(steps_generated[batch_idx]), len(steps_original[batch_idx]),
                            f"Number of steps in batch {batch_idx} differs")
            
            # Check that the steps are identical
            for step_idx in range(len(steps_generated[batch_idx])):
                self.assertEqual(steps_generated[batch_idx][step_idx], 
                               steps_original[batch_idx][step_idx],
                               f"Step {step_idx} in batch {batch_idx} differs")

    def test_student_teacher_approximation_equivalence(self):
        """Test that both implementations produce the same student-teacher approximation."""
        # First get teacher trajectory
        traj_tea = get_teacher_trajectory(
            self.model,
            self.z,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.model_kwargs
        )

        # Initialize DP tables
        dp = [torch.ones(self.teacher_steps+1, device=self.device)*torch.inf 
              for _ in range(self.student_steps+1)]
        tracestep = [torch.ones(self.teacher_steps+1, device=self.device, dtype=torch.long)*self.teacher_steps 
                    for _ in range(self.student_steps+1)]
        z_prev = torch.cat([self.z]*(self.teacher_steps+1), dim=0)
        z_next = z_prev.clone()

        # Call both implementations
        dp_generated, tracestep_generated, z_next_generated = student_teacher_approximation(
            self.model,
            z_prev,
            z_next,
            dp,
            tracestep,
            traj_tea,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.student_steps,
            self.model_kwargs
        )
        
        dp_original, tracestep_original, z_next_original = student_teacher_approximation_ref(
            self.model,
            z_prev,
            z_next,
            dp,
            tracestep,
            traj_tea,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.student_steps,
            self.model_kwargs
        )

        # Check that all lists have the same length
        self.assertEqual(len(dp_generated), len(dp_original),
                         "DP tables have different lengths")
        self.assertEqual(len(tracestep_generated), len(tracestep_original),
                         "Trace steps have different lengths")
        
        # Check each element in the lists
        for i in range(len(dp_generated)):
            self.assertTrue(torch.allclose(dp_generated[i], dp_original[i]),
                           f"DP table at index {i} differs")
            self.assertTrue(torch.allclose(tracestep_generated[i], tracestep_original[i]),
                           f"Trace step at index {i} differs")
        
        # Check z_next values
        self.assertTrue(torch.allclose(z_next_generated, z_next_original),
                        "Next z values differ in student-teacher approximation")

    def test_optimal_trajectory_equivalence(self):
        """Test that both implementations produce the same optimal trajectory."""
        # Create test inputs
        tracestep = [torch.ones(self.teacher_steps+1, device=self.device, dtype=torch.long)*self.teacher_steps 
                    for _ in range(self.student_steps+1)]
        
        # Call both implementations
        steps_generated = retrieve_optimal_trajectory(tracestep, self.student_steps)
        steps_original = retrieve_optimal_trajectory_ref(tracestep, self.student_steps)

        # Check that the steps are identical
        self.assertEqual(len(steps_generated), len(steps_original),
                         "Number of steps in optimal trajectory differs")
        
        for step_idx in range(len(steps_generated)):
            self.assertEqual(steps_generated[step_idx], steps_original[step_idx],
                            f"Step {step_idx} in optimal trajectory differs")

    def test_oss_equivalence(self):
        """Test that both implementations produce the same OSS steps."""
        # Call both implementations
        steps_generated = search_OSS_generated(
            self.model,
            self.z,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.student_steps,
            self.model_kwargs
        )
        
        steps_original = search_OSS_original(
            self.model,
            self.z,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            self.student_steps,
            self.model_kwargs
        )

        # Check that both implementations return the same number of steps
        self.assertEqual(len(steps_generated), len(steps_original),
                         "Number of batches in output differs")
        
        # Check each batch's steps
        for batch_idx in range(len(steps_generated)):
            self.assertEqual(len(steps_generated[batch_idx]), len(steps_original[batch_idx]),
                            f"Number of steps in batch {batch_idx} differs")
            
            # Check that the steps are identical
            for step_idx in range(len(steps_generated[batch_idx])):
                self.assertEqual(steps_generated[batch_idx][step_idx], 
                               steps_original[batch_idx][step_idx],
                               f"Step {step_idx} in batch {batch_idx} differs")

    def test_oss_with_different_inputs(self):
        """Test equivalence with different input shapes and parameters."""
        # Test with different batch size
        z_small = torch.randn(1, self.channels, self.height, self.width)
        class_emb_small = torch.randn(1, 512)
        
        steps_generated = search_OSS_generated(
            self.model,
            z_small,
            1,
            class_emb_small,
            self.device,
            self.teacher_steps,
            self.student_steps,
            self.model_kwargs
        )
        
        steps_original = search_OSS_original(
            self.model,
            z_small,
            1,
            class_emb_small,
            self.device,
            self.teacher_steps,
            self.student_steps,
            self.model_kwargs
        )
        
        self.assertEqual(len(steps_generated), len(steps_original))
        self.assertEqual(len(steps_generated[0]), len(steps_original[0]))

        # Test with different student steps
        steps_generated = search_OSS_generated(
            self.model,
            self.z,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            3,  # Different student steps
            self.model_kwargs
        )
        
        steps_original = search_OSS_original(
            self.model,
            self.z,
            self.batch_size,
            self.class_emb,
            self.device,
            self.teacher_steps,
            3,  # Different student steps
            self.model_kwargs
        )
        
        self.assertEqual(len(steps_generated), len(steps_original))
        for batch_idx in range(len(steps_generated)):
            self.assertEqual(len(steps_generated[batch_idx]), 3)  # Should have 3 steps


if __name__ == '__main__':
    unittest.main() 