import logging
import time
import arg
from models.api_base import APIBase
from call_api import call_api
import utils
import os
from tqdm import tqdm
from prompt import ComprehensiveStudy,gen_prompt
if __name__ == '__main__':
    current_time = int(time.time())
    logging.basicConfig(filename='./logs/{}.log'.format(current_time),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    args = arg.get_command_line_args()
    direc_name=f'./result/{args.dataset}_SNS_{args.mode}_{args.k}_input_{current_time}'
    if not os.path.exists(direc_name):
            os.makedirs(direc_name)
    input_filename = f"{direc_name}/input.json"
    output_filename = f"{direc_name}/output.json"

    result=[]
    study = ComprehensiveStudy(args.dataset,args.k)
    study.neighbor_zero_gloden_3hop_simsce(filename=input_filename,neighbor_file="/Users/ruili/Documents/aML/aGLMcode/neighbor_dict/cora_human_neighbor_zero_shot_gloden_3hop_simsce_top1_classification_input_0_1700973443.8433018.json")
    dataset=utils.read_jsonl(input_filename)


    prompt=gen_prompt(dataset,args.dataset)


    end=len(prompt)
    result=[]
    abnormal=0
    for idx in tqdm(range(0,end)):
        print(f"idx: {idx}")
        out=call_api(prompt[idx],temperature=0,max_tokens=300)
        result.append(out)
        utils.save_results(result,output_filename)

    study.eval_dataset(utils.read_output(output_filename))







   