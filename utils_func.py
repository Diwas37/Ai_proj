from diffusers import DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def create_scheduler():
    #ddpm 2M karras
    return DPMSolverMultistepScheduler(
        num_train_timesteps = 1000,
        beta_start = 0.0001,
        beta_end = 0.02,
        beta_schedule="linear",
        algorithm_type = "dpmsolver++",
        solver_order=2, 
        use_karras_sigmas = True
    )
    
def translate_to_eng(prompt):
    model_name = "VietAI/envit5-translation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
    inputs = ["vi:" + prompt]
    outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
    tran =  tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    tran = tran.replace('en: ', '')
    return tran

if __name__ == "__main__":
    prompt = "Hãy tạo một bức tranh với chất lượng cao, ánh sáng tốt, sang trọng"
    print(translate_to_eng(prompt))