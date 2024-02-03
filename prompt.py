import utils
from llmutils.utils import load_data
import dgl,torch,os,json
import numpy as np
import ast
import editdistance
def topk_accuracy(pred_texts, gt, label_names, topk = True, need_clean = True):
    error_list = []
    correct = 0
    miss = 0
    label_names = [x.lower() for x in label_names]
    for i, t in enumerate(pred_texts):
        if need_clean:
            clean_t = t.replace('.', ' ')
            clean_t = clean_t.lower()
            clean_t = clean_t.replace('\\', '')
            clean_t = clean_t.replace('_', ' ')
        else:
            clean_t = t
        # import ipdb; ipdb.set_trace()
        try:
            start = clean_t.find('[')
            end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
            result = ast.literal_eval(list_str)

            # import ipdb; ipdb.set_trace()
            res = result[0]
            if res in label_names:
                this = label_names.index(res)
                if this == gt[i]:
                    correct += 1
                    continue
                else:
                    error_list.append(i)
            else:
                miss += 1
                edits = np.array([editdistance.eval(res, l) for l in label_names])
                this = np.argmin(edits)
                if this == gt[i]:
                    correct += 1
                    continue
                else:
                    error_list.append(i)
            
        except Exception:
            miss += 1
            for k, l in enumerate(label_names):
                if l.lower() in clean_t:
                    if k == gt[i]:
                        correct += 1
                    else:
                        error_list.append(i)
                    break
    print(miss)
    return correct / len(pred_texts)


def prompt_neighbor_zero_shot_gloden(data_obj,text,dataset_name, sampled_test_node_idxs, instruction_format = 'arxiv cs xx', mapping = None, neighbor=None):
    label_names =utils.get_label(dataset_name) 
    if "arxiv" in instruction_format:
        label_names = [utils.transform_category(x) for x in label_names]
    if mapping != None:
        human_label_names = [mapping[key] for key in label_names]
    data_y = data_obj.y.numpy()
    is_product= dataset_name=="product"
    prompts = []
    for t in range(len(sampled_test_node_idxs)):

        if dataset_name=="cora":
            prompt = "{}\n".format(text['title'][sampled_test_node_idxs[t]])+"{}\n".format(text['abs'][sampled_test_node_idxs[t]])
        elif dataset_name=="pubmed":
            prompt = "Title: {}\n".format(text['title'][sampled_test_node_idxs[t]])+"Abstract: {}\n".format(text['abs'][sampled_test_node_idxs[t]])
        elif  dataset_name=="arxiv":
            prompt = "Abstract: {}\n".format(text['abs'][sampled_test_node_idxs[t]])+"Title: {}\n".format(text['title'][sampled_test_node_idxs[t]])
        elif dataset_name=='citeseer':
            prompt = "{}\n".format(text['text'][sampled_test_node_idxs[t]])
        elif dataset_name=="product":
            prompt = "Title: {}\n".format(text['title'][sampled_test_node_idxs[t]])+"Content: {}\n".format(text['content'][sampled_test_node_idxs[t]])
        else:
            raise NotImplementedError
         

        

        prompt+="\nIt has following important neighbors which has citation relationship to this paper, from most related to least related:\n"
        cur_neighbor=neighbor[t]
        for idx,j in enumerate(cur_neighbor):
            if dataset_name=="cora":
                prompt+=f"Neighbor Paper{idx}:"+" {{\n"
                if data_obj.train_mask[j] or data_obj.val_mask[j]:
                    prompt+=f"Category: {text['label'][j]}"+"\n"
                prompt+=text['title'][j]+"}}\n"
            elif dataset_name=="pubmed" or dataset_name=="arxiv":
                prompt+=f"Neighbor Paper{idx}:"+" {{\n"+"Title: "+text['title'][j]+"\n"
                if data_obj.train_mask[j] or data_obj.val_mask[j]:
                    prompt+=f"Category: {text['label'][j]}"+"}}\n"
            elif dataset_name=="citeseer":
                prompt+=f"Neighbor Paper{idx}:"+" {{\nTitle: "+utils.extract_first_n_words(text['text'][j],20)+"\n"
                if data_obj.train_mask[j] or data_obj.val_mask[j]:
                    prompt+= f"Category: {mapping[text['label'][j]]}"+"}}\n"
            elif dataset_name=="product":
                prompt+=f"Neighbor Product{idx}:"+" {{\n"+"Title: "+text['title'][j]+"\n"
                if data_obj.train_mask[j] or data_obj.val_mask[j]:
                    prompt+=f"Category: {text['label'][j]}"+"}}\n"
            else:
                raise NotImplementedError
            prompt+="\n"
        if dataset_name!="arxiv":
        
            if mapping != None:
                prompt += "Task: \n"
                prompt += "There are following categories: \n"
                prompt += (str(human_label_names) + "\n")
        
            if not is_product:
                prompt += "Which category does this paper belong to?\n"

        
            
            if dataset_name=="cora":
                prompt += f"Please comprehensively consider the information from the categories of the neighbors, and output the most 1 possible category of this paper. Please output in the form: Category: ['category']"
            elif dataset_name=="pubmed":
                prompt += f"Please comprehensively consider the information the information from the title, abstract and neighbors, and do not give any reasoning process. Output the most 1 possible category of this paper as a python list and in the form Category: ['{instruction_format}']"
            elif dataset_name=="citeseer": 
                prompt += f"Please comprehensively consider the information from the article and its neighbors, and output the most 1 possible category of this paper as a python list and in the form Category: ['{instruction_format}']"

            elif is_product:
                prompt+="\nPlease predict the most likely category of this product from Amazon. Please output in the form ['your category']."
            else:
                raise NotImplementedError
        else:
            prompt+="\n\nPlease comprehensively consider the information from the categories of the neighbors and predict the most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format ['cs.XX']."

        prompts.append(prompt)
    if mapping != None:
        return prompts, human_label_names
    else:
        return prompts





class ComprehensiveStudy:
    def __init__(self,dataset,k):
        self.datasets = dataset
        self.arxiv_mapping, self.citeseer_mapping, self.pubmed_mapping, self.cora_mapping, self.products_mapping = utils.load_mapping()
        self.split = "fixed"
        self.seeds = 0
        self.sample_num = 1000
        self.mapping, self.dataset_graph,self.text, self.sampled_test_node_idxs, self.train_node_idxs = self.prepare_dataset(self.datasets, self.split, self.seeds)
        self.k=utils.get_k(k,dataset)

    def prepare_dataset(self, dataset_name, split, seed):
        utils.set_seed_config(seed)
        dataset, text = load_data(dataset_name, use_text=True, seed=seed)
        sample_num = self.sample_num
        sampled_test_node_idxs, train_node_idxs = utils.get_sampled_nodes(dataset, sample_num,dataset_name)

        print(f"{dataset_name} data processed!")
        instruction = 'XX'
        if dataset_name == "arxiv":
            mapping = self.arxiv_mapping
        elif dataset_name == 'citeseer':
            mapping = self.citeseer_mapping
        elif dataset_name == 'pubmed':
            mapping = self.pubmed_mapping
        elif dataset_name == 'cora':
            mapping = self.cora_mapping
        elif dataset_name == 'product':
            mapping = self.products_mapping

        return mapping, dataset,text, sampled_test_node_idxs, train_node_idxs
    def get_graph(self,data):
        g = dgl.graph((data["edge_index"][0], data["edge_index"][1]), num_nodes=data.num_nodes)
        g = dgl.remove_self_loop(g)
        g=dgl.to_bidirected(g)

        g = dgl.add_self_loop(g)
        device = torch.device("cpu")
        g = g.int().to(device)
        g.ndata["feat"]=data['x']
        features = g.ndata["feat"]
        g.ndata["label"]=data['y']
        labels = g.ndata["label"]
        return g
    
    def neighbor_zero_gloden_3hop_simsce(self, filename=None,instruction='XX',neighbor_file=None):
        mapping,dataset,text, sampled_test_node_idxs, train_node_idxs, seed, dataset_name=self.mapping, self.dataset_graph,self.text, self.sampled_test_node_idxs, self.train_node_idxs,self.seeds,self.datasets

        if neighbor_file == None:
            graph=self.get_graph(dataset)
            combine_text=utils.get_combine_text(text,dataset_name)

            neighbor_list=utils.get_top_k_3hop_neighbor_with_label_simcse(graph,combine_text,sampled_test_node_idxs,k=100,train_mask=dataset.train_mask,val_mask=dataset.val_mask)
            utils.save_neighbor(neighbor_list,dataset_name,seed)

            neighbor_list=utils.load_neighbor(f"./neighbor_dict/{dataset_name}_neighbor_{seed}.json")
        else:
            neighbor_list=utils.load_neighbor(neighbor_file)

        
        neighbor=[i[:self.k] for i in neighbor_list]

        zero_shot_prompt, human_labels = prompt_neighbor_zero_shot_gloden(dataset,text,dataset_name, sampled_test_node_idxs, instruction_format=instruction, mapping = mapping,neighbor=neighbor)
        # import ipdb; ipdb.set_trace()
       
        jobs = utils.generate_chat_input_file(zero_shot_prompt)
        with open(filename, "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")

    def eval_dataset(self,output):
        mapping,dataset,text, sampled_test_node_idxs, train_node_idxs, seed, dataset_name=self.mapping, self.dataset_graph,self.text, self.sampled_test_node_idxs, self.train_node_idxs,self.seeds,self.datasets
        y,human_labels=utils.get_golden(text,dataset_name, sampled_test_node_idxs, mapping = mapping)
        top1_acc = topk_accuracy(output, y, human_labels, topk = False)
        print(f"{dataset_name} accuracy: {top1_acc}")


def gen_prompt(dataset, datasetname):
    out=[]
    for idx,i in enumerate(dataset):
        out.append( {"idx":idx,"prompt":[{'role': 'system', 'content':utils.system_prompt_sns[datasetname]},{'role': 'user', 'content':i['input']}]})
    return out