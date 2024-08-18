from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertModel, BertTokenizer
import torch
import numpy as np

# Load pre-trained models and tokenizers
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
gpt_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

def extractive_summarization(text, num_sentences=10):
    # Tokenize and encode sentences
    sentences = text.split('. ')
    inputs = bert_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

    # Get sentence embeddings without needing to explicitly pass attention_mask
    with torch.no_grad():
        outputs = bert_model(**inputs)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

    # Rank sentences by their embeddings (simplified with random scores for demonstration)
    sentence_scores = np.random.rand(len(sentences))
    top_sentence_indices = sentence_scores.argsort()[:num_sentences]
    top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
    return '. '.join(top_sentences)

def abstractive_summarization(text):
    inputs = gpt_tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = gpt_model.generate(
        inputs,
        max_new_tokens=150,  # Controls the length of the generated summary
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=gpt_tokenizer.eos_token_id
    )
    summary = gpt_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def rule_based_additions(original_text, summary_text):
    critical_sections = ["allergies", "medications", "critical past medical history"]
    additional_info = []
    for section in critical_sections:
        if section in original_text.lower() and section not in summary_text.lower():
            start_index = original_text.lower().find(section)
            end_index = original_text.lower().find('.', start_index)
            additional_info.append(original_text[start_index:end_index + 1])
    return summary_text + ' '.join(additional_info)

# Example patient history
patient_history = """
Patient Name: John Doe
Date of Birth: January 1, 1980
Chief Complaint: Persistent cough for 2 weeks.
History of Present Illness: Started 2 weeks ago, continuous dry cough. Aggravated by cold air and physical exertion. Relieved by rest and warm fluids. Associated with mild fever and fatigue.
Past Medical History: Hypertension (2010), Type 2 Diabetes (2015), Appendectomy (2005).
Medications: Metformin 500 mg BID, Lisinopril 10 mg daily, Atorvastatin 20 mg daily.
Allergies: Penicillin (rash).
Family History: Father: Hypertension, died at 70 (heart attack). Mother: Type 2 Diabetes, alive at 68.
Social History: Non-smoker, occasional alcohol use, teacher, lives with spouse and two children.
Review of Systems: General: Fatigue, no weight changes. Respiratory: Dry cough, no shortness of breath. Cardiovascular: No chest pain, no palpitations. Gastrointestinal: No nausea, vomiting, or diarrhea.
"""

# Perform extractive summarization
extractive_summary = extractive_summarization(patient_history)

# Perform abstractive summarization on the extractive summary
abstractive_summary = abstractive_summarization(extractive_summary)

# Ensure rule-based critical information is included
final_summary = rule_based_additions(patient_history, abstractive_summary)

print("Final Summary:\n", final_summary)