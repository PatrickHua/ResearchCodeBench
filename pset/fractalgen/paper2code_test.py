

from models import fractalgen
import torch
# import numpy as np
import os
import pickle
import argparse
import time

# from util import download


class TestFractalGenSampler:

    def __init__(self):
        self.seed = 0 #@param {type:"number"}
        # Set seeds
        torch.manual_seed(self.seed)


        
        # self.function_output_dir = "paper2code_test_output"
        # os.makedirs(self.function_output_dir, exist_ok=True)
        # self.function_output_file = os.path.join(self.function_output_dir, "sampled_images.pkl")
        self.function_output_file = "sampled_images.pkl"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise NotImplementedError("CPU not supported")
        self.device = device
        model_type = "fractalmar_in64_paper2code"
        if model_type == "fractalar_in64_paper2code":
            num_conds=1
        elif model_type == "fractalmar_in64_paper2code":
            num_conds=5
        else:
            raise NotImplementedError

        self.model = fractalgen.__dict__[model_type](
            guiding_pixel=False,
            num_conds=num_conds,
        ).to(device)

        self.num_conds = num_conds
        self.model.eval()
        
        # Set user inputs

        self.class_labels = 207, 360, 388, 113, 355, 980, 323, 979 #@param {type:"raw"}
        self.num_iter_list = 16, 4 #@param {type:"raw"}
        self.cfg_scale = 5 #@param {type:"slider", min:1, max:10, step:0.5}
        self.cfg_schedule = "constant" #@param ["linear", "constant"]
        self.temperature = 1.02 #@param {type:"slider", min:0.9, max:1.2, step:0.01}
        self.filter_threshold = 1e-4
        self.samples_per_row = 4 #@param {type:"number"}

        self.label_gen = torch.Tensor(self.class_labels).long().to(device)
        self.class_embedding = self.model.class_emb(self.label_gen)
        if not self.cfg_scale == 1.0:
            self.class_embedding = torch.cat([self.class_embedding, self.model.fake_latent.repeat(self.label_gen.size(0), 1)], dim=0)


    def forward_fractalgen_sampler(self):
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                torch.manual_seed(self.seed)
                sampled_images = self.model.sample(
                    cond_list=[self.class_embedding for _ in range(self.num_conds)],
                    num_iter_list=self.num_iter_list,
                    cfg=self.cfg_scale, cfg_schedule=self.cfg_schedule,
                    temperature=self.temperature,
                    filter_threshold=self.filter_threshold,
                    fractal_level=0,
                    visualize=False)
        return sampled_images


    def sanity_check_fractalgen_sampler(self):
        # check deterministic
        sampled_images = self.forward_fractalgen_sampler()
        sampled_images_2 = self.forward_fractalgen_sampler()

        assert torch.allclose(sampled_images, sampled_images_2)

        with open(self.function_output_file, "wb") as f:

            pickle.dump(sampled_images, f)
        
        
        
    def test_check_fractalgen_sampler_equivalence(self):

        try:
            sampled_images = self.forward_fractalgen_sampler()
        except Exception as e:
            print(f"Error: {e}")
            return False
            
        with open(self.function_output_file, "rb") as f:
            expected_output = pickle.load(f)
        # breakpoint()
        difference = torch.abs(sampled_images - expected_output)
        print(difference.sum())
        # breakpoint()
        if torch.allclose(sampled_images, expected_output, atol=1e-4):
            print("All close")
            return True
        else:
            print("Not all close")
            return False
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_check", action="store_true")
    # parser.add_argument("--test_equivalence", action="store_true")
    args = parser.parse_args()
    
    test_fractalgen_sampler = TestFractalGenSampler()
    if args.sanity_check:
        test_fractalgen_sampler.sanity_check_fractalgen_sampler()
        # test_fractalgen_sampler.test_check_fractalgen_sampler_equivalence()
    # if args.test_equivalence:
        # breakpoint()
        # test_fractalgen_sampler.sanity_check_fractalgen_sampler()
        # time the test_check_fractalgen_sampler_equivalence
    start_time = time.time()
    test_fractalgen_sampler.test_check_fractalgen_sampler_equivalence()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

