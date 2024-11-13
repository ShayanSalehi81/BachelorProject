import re
import torch
import warnings
import pandas as pd
import bitsandbytes

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings('ignore')
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


class Generator:
    def __init__(self, model_name, quantize_4bit=True, use_flash_attention=False):
        self.model_name = model_name
        self.quantize_4bit = quantize_4bit
        self.use_flash_attention = use_flash_attention
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        quantization_config = None
        if self.quantize_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        attn_implementation = None
        if self.use_flash_attention:
            attn_implementation = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Model and tokenizer loaded successfully.")

    def get_message_format(self, system_prompt, user_prompts):
        formatted_prompts = []
        for user_prompt in user_prompts:
            formatted_prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
        return formatted_prompts

    def generate_responses(self, system_prompt, user_prompts, temperature=0.3, top_p=0.75, top_k=0, max_new_tokens=1024):
        messages = self.get_message_format(system_prompt, user_prompts)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_padded_len = len(input_ids[0])
        gen_tokens = self.model.generate(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
        gen_tokens = [gt[prompt_padded_len:] for gt in gen_tokens]
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)


class LLMRunner():
    def __init__(self, train_path, eval_path, test_path, result_path, base_prompt_path, kshot_prompt_path, output_path, kshot_list):
        self.train_df, self.eval_df, self.test_df, self.result_df = self.load_datasets(train_path, eval_path, test_path, result_path)
        self.base_prompt, self.kshot_prompt = self.load_prompts(base_prompt_path, kshot_prompt_path)
        self.kshot_list = kshot_list
        self.output_path = output_path
        self.model_name = 'CohereForAI/aya-23-8B'
        self.generator = Generator(self.model_name)
        self.main_dataframe = self.train_df
        self.final_classification_text = 'Final Classification:'
        self.example = 'Example:'

    def load_datasets(self, train_path, eval_path, test_path, result_path):
        train_df = pd.read_excel(train_path)
        eval_df = pd.read_excel(eval_path)
        test_df = pd.read_excel(test_path)
        result_df = pd.read_csv(result_path)
        return train_df, eval_df, test_df, result_df
    
    def load_prompts(self, base_path, kshot_path):
        with open(base_path, 'r', encoding='utf-8') as f:
            base_prompt = f.read()
        with open(kshot_path, 'r', encoding='utf-8') as f:
            kshot_prompt = f.read()
        return base_prompt, kshot_prompt

    def get_label(self, text):
        if text == 1:
            return "1"
        elif text == 0:
            return "0"
    
    def extract_label(self, llm_output):
        match = re.search((self.final_classification_text + r'\s*(\d)\s*'), llm_output)
        if match:
            return int(match.group(1))
        else:
            return "0"
        
    def extract_first_number(self, input_string):
        match = re.search(r'\d+', input_string)
        if match:
            return int(match.group(0))
        else:
            return None
        
    def get_k_most_similar_texts_by_tfidf(self, df, target_text, texts=None, k=5):
        texts = []
        for _, row in df.iterrows():
            texts.append((row[1], row[2], row[4]))

        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        text_vectors = vectorizer.fit_transform([text[0] for text in texts] + [target_text])

        cosine_similarities = cosine_similarity(text_vectors[-1], text_vectors[:-1])
        cosine_similarities = cosine_similarities[0]  # Extract the first row from the 2D array

        top_indices = cosine_similarities.argsort()[::-1][:k]

        results = [(texts[i][0], self.get_label(texts[i][-1]), cosine_similarities[i]) for i in top_indices]
        return results
    
    def generate_results(self, k_shot):
        column_name_to_write = f'predicted_k_{k_shot}'
        start_row = self.result_df.index[pd.isna(self.result_df[column_name_to_write])].tolist()[0]
        
        print(f'K Shot learning = {k_shot}, Start row = {start_row}')

        for i in range(start_row, len(self.result_df)):
            if k_shot == 0:
                prompt_fa_kshot = self.base_prompt
            else:
                prompt_fa_kshot = self.kshot_prompt

            result_df_counter = i % len(self.result_df)
            print(f"result_df_counter is {result_df_counter}")

            target_text = self.result_df['text'][i]
            if (len(target_text) > 10000):
                target_text = target_text[:8000]
                
            new_prompt = prompt_fa_kshot
            user_prompt = target_text

            if k_shot != 0:
                sample_str = ''
                for _ in range(k_shot):
                    sample_str += self.example + ' {}\n' + '{}'

                new_prompt = new_prompt.replace('SAMPLES_HERE', sample_str)
                samples = []
                similar_texts = self.get_k_most_similar_texts_by_tfidf(self.train_df, self.main_dataframe['title'][result_df_counter] + '\n' + self.main_dataframe['text'][result_df_counter], k=k_shot)

                for text in similar_texts:
                    samples.append(text[0] + ' ' + self.final_classification_text + ' ' + text[1])
                    samples.append('\n')

                new_prompt = new_prompt.format(*samples)
                
            system_prompt = new_prompt
            user_prompt_list = [user_prompt]

            output = self.generator.generate_responses(system_prompt, user_prompt_list)[0]
            
            result = self.extract_label(output)
            self.result_df.at[i, column_name_to_write] = result
            
            torch.cuda.empty_cache()
            print(f"answer of row {i} is {result} and k is {k_shot}.     Text type: {self.result_df['text_type'][i]}  Real tag: {self.result_df['real_tag'][i]}\nOutput: {output}")

            if i % 20 == 0:
                self.result_df.to_csv(self.output_path, index=False)
                print(f"dataframe saved to csv file at iteration {i}")

        self.result_df.to_csv(self.output_path, index=False)
        print("dataframe saved to csv file at final iteration")

    def run(self):
        for k_shot in self.kshot_list:
            self.generate_results(k_shot)
    

if __name__ == '__main__':
    llm_runner = LLMRunner(
        train_path='/kaggle/input/news-dataset/train_bert_cat.xlsx',
        eval_path='/kaggle/input/news-dataset/eval_bert_cat.xlsx',
        test_path='/kaggle/input/news-dataset/test_bert_cat.xlsx',
        result_path='/kaggle/input/news-dataset/System_Prompt19_Train_Data_Results.csv',
        base_prompt_path='/kaggle/input/news-prompt/base_system_prompt_19.txt',
        kshot_prompt_path='/kaggle/input/news-prompt/kshot_system_prompt_19.txt',
        output_path='System_Prompt19_Train_Data_Results.csv',
        kshot_list=[0])
    llm_runner.run()