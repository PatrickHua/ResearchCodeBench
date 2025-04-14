import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import PowerMeanNoise_PerColumn, UnifiedCtimeDiffusion

class TestNoiseRateRho(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_initialization(self):
        """Test that PowerMeanNoise_PerColumn initializes with correct parameters."""
        num_numerical = 5
        rho_init = 1.0
        rho_offset = 2.0
        
        # Create noise schedule
        noise_schedule = PowerMeanNoise_PerColumn(
            num_numerical=num_numerical,
            rho_init=rho_init,
            rho_offset=rho_offset
        )
        
        # Check initialization values
        self.assertEqual(noise_schedule.num_numerical, num_numerical)
        self.assertEqual(noise_schedule.rho_offset, rho_offset)
        
        # Check that rho_raw is initialized correctly
        self.assertEqual(noise_schedule.rho_raw.shape, torch.Size([num_numerical]))
        self.assertTrue(torch.all(noise_schedule.rho_raw == rho_init))
        
        # Check that rho() returns values greater than rho_offset
        rho_values = noise_schedule.rho()
        self.assertTrue(torch.all(rho_values > rho_offset))
        
    def test_rho_transformation(self):
        """Test that the softplus transformation of rho works correctly."""
        num_numerical = 3
        rho_init = 1.0
        rho_offset = 2.0
        
        noise_schedule = PowerMeanNoise_PerColumn(
            num_numerical=num_numerical,
            rho_init=rho_init,
            rho_offset=rho_offset
        )
        
        # Get rho values
        rho_values = noise_schedule.rho()
        
        # Check that rho values are transformed correctly
        expected_rho = nn.functional.softplus(torch.tensor([rho_init] * num_numerical)) + rho_offset
        self.assertTrue(torch.allclose(rho_values, expected_rho))
        
    def test_learnable_parameter(self):
        """Test that rho_raw is a learnable parameter."""
        num_numerical = 4
        noise_schedule = PowerMeanNoise_PerColumn(num_numerical=num_numerical)
        
        # Initial parameter values
        initial_rho_raw = noise_schedule.rho_raw.clone()
        
        # Create optimizer
        optimizer = torch.optim.SGD([noise_schedule.rho_raw], lr=0.1)
        
        # Custom loss function that encourages rho to be larger
        loss_fn = lambda: -noise_schedule.rho().mean()
        
        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            optimizer.step()
        
        # Parameter should have changed after training
        self.assertFalse(torch.allclose(noise_schedule.rho_raw, initial_rho_raw))
        
    def test_noise_schedule_behavior(self):
        """Test that the noise schedule behaves correctly with different rho values."""
        num_numerical = 2
        noise_schedule = PowerMeanNoise_PerColumn(num_numerical=num_numerical)
        
        # Test different time steps
        t = torch.tensor([0.0, 0.5, 1.0]).unsqueeze(-1)  # [3, 1]
        
        # Get noise values
        sigma = noise_schedule.total_noise(t)
        
        # Check shape
        self.assertEqual(sigma.shape, (3, num_numerical))  # [time_steps, num_numerical]
        
        # Check that noise increases with time
        self.assertTrue(torch.all(sigma[1] > sigma[0]))
        self.assertTrue(torch.all(sigma[2] > sigma[1]))
        
        # Check that noise values are positive
        self.assertTrue(torch.all(sigma > 0))

    def test_noise_bounds_calculation(self):
        """Test calculation of lower and upper bounds for noise sigma."""
        num_numerical = 3
        noise_schedule = PowerMeanNoise_PerColumn(num_numerical=num_numerical)
        
        rho = noise_schedule.rho()
        sigma_min = noise_schedule.sigma_min
        sigma_max = noise_schedule.sigma_max
        
        # Calculate bounds
        sigma_min_pow = sigma_min ** (1 / rho)
        sigma_max_pow = sigma_max ** (1 / rho)
        
        # Check shapes
        self.assertEqual(sigma_min_pow.shape, torch.Size([num_numerical]))
        self.assertEqual(sigma_max_pow.shape, torch.Size([num_numerical]))
        
        # Check values
        self.assertTrue(torch.all(sigma_min_pow < sigma_max_pow))
        self.assertTrue(torch.all(sigma_min_pow > 0))
        self.assertTrue(torch.all(sigma_max_pow > 0))

    def test_power_mean_noise_calculation(self):
        """Test calculation of power-mean noise sigma."""
        num_numerical = 2
        noise_schedule = PowerMeanNoise_PerColumn(num_numerical=num_numerical)
        
        t = torch.tensor([0.0, 0.5, 1.0]).unsqueeze(-1)  # [3, 1]
        rho = noise_schedule.rho()
        sigma_min = noise_schedule.sigma_min
        sigma_max = noise_schedule.sigma_max
        
        # Calculate noise
        sigma_min_pow = sigma_min ** (1 / rho)
        sigma_max_pow = sigma_max ** (1 / rho)
        sigma = (sigma_min_pow + t * (sigma_max_pow - sigma_min_pow)).pow(rho)
        sigma_llm = noise_schedule.total_noise(t)

        # Check properties
        self.assertTrue(torch.allclose(sigma, sigma_llm))

    def test_time_calculation(self):
        """Test calculation of time t from noise sigma."""
        num_numerical = 2
        noise_schedule = PowerMeanNoise_PerColumn(num_numerical=num_numerical)
        
        # Generate some sigma values
        t = torch.tensor([0.0, 0.5, 1.0]).unsqueeze(-1)  # [3, 1]
        sigma = noise_schedule.total_noise(t)
        
        # Calculate t back from sigma
        t_calculated = noise_schedule.inverse_to_t(sigma)
        
        # Check that we can recover the original t values
        self.assertTrue(torch.allclose(t_calculated, t))

    def test_gaussian_noise_addition(self):
        """Test addition of Gaussian noise to numerical features."""
        batch_size = 4
        num_numerical = 3
        x_num = torch.randn(batch_size, num_numerical)
        sigma = torch.tensor([0.1, 0.5, 1.0])
        
        # Add noise
        noise = torch.randn_like(x_num)
        x_num_t = x_num + noise * sigma
        
        # Check properties
        self.assertEqual(x_num_t.shape, x_num.shape)
        self.assertFalse(torch.allclose(x_num_t, x_num))
        self.assertTrue(torch.allclose(x_num_t - x_num, noise * sigma))

    def test_categorical_corruption(self):
        """Test corruption of categorical data."""
        batch_size = 4
        num_categories = 2
        num_classes = [1, 1]  # Two categorical features with 3 and 4 classes respectively
        x_cat = torch.tensor([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 0]
        ])
        
        # Create model
        model = UnifiedCtimeDiffusion(
            num_classes=np.array(num_classes),
            num_numerical_features=0,
            denoise_fn=lambda x, y, t, sigma: (None, None),
            y_only_model=False
        )
        noise_schedule = model.cat_schedule
        # Test hard corruption
        t = torch.rand(batch_size, 1)
        t = t[:, None]
        sigma_cat = noise_schedule.total_noise(t)
        move_chance = -torch.expm1(-sigma_cat)
        xt, xt_soft = model.q_xt(x_cat, move_chance, strategy='hard')
        
        # Check properties
        self.assertTrue(torch.all((xt == x_cat) | (xt == model.mask_index)))
        
        
if __name__ == '__main__':
    unittest.main() 