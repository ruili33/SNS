import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
device="cuda" if torch.cuda.is_available() else "cpu"
# Import our models. The package will take care of downloading the models automatically
class simcse():
    def __init__(self) -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").to(device)

    def get_sim(self,text1,text2_list=[]):
        texts=[text1]+text2_list
        # 假设 texts 是一个包含大量文本的列表
        batch_size = 32
        all_embeddings = []  # 用于存储每个批次的 embeddings

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                all_embeddings.append(embeddings)

        # 拼接所有批次的 embeddings
        concatenated_embeddings = torch.cat(all_embeddings, dim=0)
        sim_score=[]
        for i in range(len(text2_list)):
            sim_score.append((i,1 - cosine(concatenated_embeddings[0].cpu(), concatenated_embeddings[1+i].cpu())))
        return sim_score


    def return_top(self,text1,text2_list=[],orig_neighbor_index_list=[],k=0):
        sim_score=self.get_sim(text1,text2_list)
        sorted_score = sorted(sim_score, key=lambda item: item[1], reverse=True)
        if len(text2_list)>k:
            return [orig_neighbor_index_list[sorted_score[i][0]] for i in range(k)]
        else:
            return [orig_neighbor_index_list[sorted_score[i][0]] for i in range(len(text2_list))]
        

