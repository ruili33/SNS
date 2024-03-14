import torch,json,random,os
import numpy as np
from tqdm import tqdm
from mysimcse import simcse
def load_mapping():
    arxiv_mapping = {
    'cs.AI': 'Artificial Intelligence',
    'cs.CL': 'Computation and Language',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.CR': 'Cryptography and Security',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.DB': 'Databases',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.AR': 'Hardware Architecture',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LO': 'Logic in Computer Science',
    'cs.LG': 'Machine Learning',
    'cs.MS': 'Mathematical Software',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NA': 'Numerical Analysis',
    'cs.OS': 'Operating Systems',
    'cs.OH': 'Other Computer Science',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SI': 'Social and Information Networks',
    'cs.SE': 'Software Engineering',
    'cs.SD': 'Sound',
    'cs.SC': 'Symbolic Computation',
    'cs.SY': 'Systems and Control'
}
    citeseer_mapping = {
        "Agents": "Agents",
        "ML": "Machine Learning",
        "IR": "Information Retrieval",
        "DB": "Database",
        "HCI": "Human Computer Interaction",
        "AI": "Artificial Intelligence"
    }
    # pubmed_mapping = {
    #     'Diabetes Mellitus, Experimental': 'Diabetes Mellitus, Experimental',
    #     'Diabetes Mellitus Type 1': 'Diabetes Mellitus Type 1',
    #     'Diabetes Mellitus Type 2': 'Diabetes Mellitus Type 2'
    # }
    pubmed_mapping = {
        'Diabetes Mellitus, Experimental': 'Experimentally induced diabetes',
        'Diabetes Mellitus Type 1': 'Type 1 diabetes',
        'Diabetes Mellitus Type 2': 'Type 2 diabetes'
    }
    cora_mapping = {
        'Rule_Learning': "Rule Learning",
        'Case_Based': "Case Based",
        'Genetic_Algorithms': "Genetic Algorithms",
        'Theory': "Theory",
        'Reinforcement_Learning': "Reinforcement Learning",
        'Probabilistic_Methods': "Probabilistic Methods",
        'Neural_Networks': "Neural Networks"
    }

    products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}
    return arxiv_mapping, citeseer_mapping, pubmed_mapping, cora_mapping, products_mapping


def set_seed_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def read_jsonl(file_path):
    """
    Read a .jsonl file and return the contents as a list of dictionaries.

    Parameters:
    file_path (str): The path to the .jsonl file to be read.

    Returns:
    list: A list of dictionaries, each representing a JSON object.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def read_json(file_path):
    """
    Read a .jsonl file and return the contents as a list of dictionaries.

    Parameters:
    file_path (str): The path to the .jsonl file to be read.

    Returns:
    list: A list of dictionaries, each representing a JSON object.
    """
    data = []
    with open(file_path) as file:
        data=json.load(file)
    return data


def get_sampled_nodes(data_obj, sample_num = -1,datasetname=None):
    train_mask = data_obj['train_mask']
    # val_mask = data_obj.val_masks[0]
    test_mask = data_obj['test_mask']
    all_idxs = torch.arange(data_obj.x.shape[0])
    test_node_idxs = all_idxs[test_mask]
    train_node_idxs = all_idxs[train_mask]
    # val_node_idxs = all_idxs[val_mask]
    if sample_num == -1:
        sampled_test_node_idxs = test_node_idxs
    else:
        sampled_test_node_idxs = test_node_idxs[torch.randperm(test_node_idxs.shape[0])[:sample_num]]

    if datasetname in ["cora", "pubmed", "citeseer"]:
        sampled_test_node_idxs = torch.sort(sampled_test_node_idxs).values
    return sampled_test_node_idxs, train_node_idxs


def get_combine_text(text,dataset_name):
    if dataset_name=="cora" or dataset_name=="pubmed" or dataset_name=="arxiv":
        title=text['title']
        abstract=text['abs']
        combine_text=[title[i]+abstract[i] for i in range(len(title))]
        return combine_text
    elif dataset_name=="citeseer":
        return text['text']
    elif dataset_name=="product":
        title=text['title']
        abstract=text['content']
        combine_text=["Title: "+title[i]+"Content: "+abstract[i] for i in range(len(title))]
        return combine_text
    else:
        raise NotImplementedError

def save_neighbor(neighbor,dataset_name,seed):
    with open(f"./neighbor_dict/{dataset_name}_neighbor_{seed}.json","w") as file:
        json.dump({"neighbor":neighbor},file)
def load_neighbor(filename):
    with open(filename) as file:
        neighbor_dict=json.load(file)
    return neighbor_dict["neighbor"]

def get_k_hop_neighbors(g, node_idx, k):
    """
    Get the k-hop neighbors of a given node in the graph.

    Args:
    - g (dgl.DGLGraph): The graph.
    - node_idx (int or list): Index of the node(s) for which to find k-hop neighbors.
    - k (int): The number of hops.

    Returns:
    - list: List of k-hop neighbor node indices.
    """
    visited = set([node_idx])
    current_level = set([node_idx])
    for _ in range(k):
        next_level = set()
        for u in current_level:
            if isinstance(u,torch.Tensor):
                u=u.item()
            _, successors = g.out_edges(u, form='uv')
            # successors_in,_=g.in_edges(u, form='uv')
            for v in successors.tolist():
                if v not in visited:
                    visited.add(v)
                    next_level.add(v)
        current_level = next_level
    visited=list(visited)
    visited.remove(node_idx)
    
    for idx,i in enumerate(visited):
        if isinstance(i,torch.Tensor):
            visited[idx]=i.item()
    return visited

def get_top_k_neighbor_with_label_simcse(graph,text, sampled_test_node_idxs,k,train_mask,val_mask):
    out=[]
    simcsemoddel=simcse()
    for i in tqdm(sampled_test_node_idxs):
        neighbors=get_k_hop_neighbors(graph,i,1)
        with_label_list=[]
        without_label_list=[]
        for node_index in neighbors:
            if train_mask[node_index] or val_mask[node_index]:
                with_label_list.append(node_index)
            else:
                without_label_list.append(node_index)
        if len(with_label_list)<2:
            neighbors=get_k_hop_neighbors(graph,i,2)
            with_label_list=[]
            without_label_list=[]
            for node_index in neighbors:
                if train_mask[node_index] or val_mask[node_index]:
                    with_label_list.append(node_index)
                else:
                    without_label_list.append(node_index)
            if len(with_label_list)<1:
                neighbors=get_k_hop_neighbors(graph,i,3)
                with_label_list=[]
                without_label_list=[]
                for node_index in neighbors:
                    if train_mask[node_index] or val_mask[node_index]:
                        with_label_list.append(node_index)
                    else:
                        without_label_list.append(node_index)
                if len(with_label_list)<1:
                    neighbors=get_k_hop_neighbors(graph,i,4)
                    with_label_list=[]
                    without_label_list=[]
                    for node_index in neighbors:
                        if train_mask[node_index] or val_mask[node_index]:
                            with_label_list.append(node_index)
                        else:
                            without_label_list.append(node_index)
                    if len(with_label_list)<1:
                        neighbors=get_k_hop_neighbors(graph,i,5)
                        with_label_list=[]
                        without_label_list=[]
                        for node_index in neighbors:
                            if train_mask[node_index] or val_mask[node_index]:
                                with_label_list.append(node_index)
                            else:
                                without_label_list.append(node_index)
        cur_out=simcsemoddel.return_top(text[i],[text[j] for j in with_label_list],with_label_list,k)   
        out.append(cur_out)
    return out

def get_k(k,dataset_name):
    if k >0:
        return k
    else:
        if dataset_name=="product":
            return 100
        elif dataset_name=="citeseer":
            return 8
        else:
            return 4
        

def generate_chat_input_file(input_text, model_name = 'gpt-3.5-turbo'):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['input'] = text
        jobs.append(obj)
    return jobs 

def get_label(dataset):
    arxiv_mapping, citeseer_mapping, pubmed_mapping, cora_mapping, products_mapping=load_mapping()
    if dataset=='cora':
        mapp=cora_mapping
    elif dataset=='citeseer':
        mapp=citeseer_mapping
    elif dataset=="pubmed":
        mapp=pubmed_mapping
    elif dataset=="arxiv":
        mapp=arxiv_mapping
    elif dataset=="product":
        mapp=products_mapping
    else:
        raise NotImplementedError
    return list(mapp.keys())

def transform_category(category):
    parts = category.split()
    if len(parts) != 3 or parts[0].lower() != 'arxiv' or parts[1].lower() != 'cs':
        raise ValueError("Input should be in the format 'arxiv cs xx'")
    return "{} {}.{}".format(parts[0], parts[1], parts[2].upper())


def extract_first_n_words(text,n):
    """
    Extract the first 10 words from a given text.

    :param text: String from which to extract words.
    :return: String containing the first 10 words or the entire text if it has less than 10 words.
    """
    words = text.split()  # Split the text into words
    return ' '.join(words[:n])  # Join and return the first 10 words

def save_results(result,filename):
    with open(filename,"w") as file:
        json.dump(result, file, ensure_ascii=False, indent=2) 

def get_golden(text,dataset_name, sampled_test_node_idxs,  mapping = None):
    label_names =get_label(dataset_name) 
    if dataset_name in ['citeseer']:
        human_label_names=[key for key in label_names]
        human_label_names_true = [mapping[key] for key in label_names]
    elif dataset_name in ['arxiv']:
        human_label_names=[key for key in label_names]
        human_label_names_true=human_label_names
    else:
        human_label_names = [mapping[key] for key in label_names]
        human_label_names_true = [mapping[key] for key in label_names]
    label=text['label']
    out=[]
    index_list=[]
    for i in sampled_test_node_idxs:
        cur_label=label[i]
        cur_idx=human_label_names.index(cur_label)
        index_list.append(cur_idx)
    return index_list,human_label_names_true


def read_output(filename):
    with open(filename) as file:
        out=json.load(file)

    output=[i['result'][0] for i in out]
    return output





system_prompt_sns={
    "cora":"There are following categories: \n['Rule Learning', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods', 'Neural Networks']\nWhich category does this paper belong to?\nPlease comprehensively consider the information from the categories of the neighbors, and output the most 1 possible category of this paper. Please output in the form: Category: ['category'].",
    "pubmed":"Task: \nThere are following categories: \n['Experimentally induced diabetes', 'Type 1 diabetes', 'Type 2 diabetes']\nWhich category does this paper belong to?\nPlease comprehensively consider the information the information from the title, abstract and neighbors, and do not give any reasoning process. Output the most 1 possible category of this paper as a python list and in the form Category: ['XX'].",
    "citeseer":"Task: \nThere are following categories: \n['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction', 'Artificial Intelligence']\nWhich category does this paper belong to?\nPlease comprehensively consider the information from the article and its neighbors, and output the most 1 possible category of this paper as a python list and in the form Category: ['XX'].",
    "arxiv":"Please comprehensively consider the information from the categories of the neighbors and predict the most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format ['cs.XX'].",
    "product":"Task: \nThere are following categories: \n['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510']\n\nPlease predict the most likely category of this product from Amazon. Please output in the form ['your category']."
}