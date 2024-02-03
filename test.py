import logging
import time
import arg
from models.api_base import APIBase
from call_api import call_api
import utils
import os
from tqdm import tqdm
from prompt import ComprehensiveStudy,gen_prompt
dataset_name="product"
study = ComprehensiveStudy(dataset_name,-1)
file_name="/Users/ruili/Documents/aML/aGLMcode/openai_in/product_human_neighbor_zero_shot_gloden_3hop_simsce_top1_classification_input_0_1702259173.848829/c.json"
study.eval_dataset(utils.read_output(file_name))