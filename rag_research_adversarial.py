import pandas as pd
import random
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_community.document_transformers import LongContextReorder
from datasets import load_dataset, Dataset, Features, Sequence, Value
from ragas import evaluate, RunConfig
from ragas.metrics import context_precision, faithfulness, answer_relevancy

print("🚀 Setting up Research Experiment: The 'Adversarial' Test...")

# 1. MODELS
generator_llm = ChatOllama(model="mistral", temperature=0)
judge_llm = ChatOllama(model="mistral", temperature=0)
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
reordering = LongContextReorder()

# 2. DATA PREP
print("   • Downloading SQuAD (v1.1)...")
squad = load_dataset("squad", split="validation")
unique_contexts = list(set(squad['context'][:1000])) # 1,000 Doc Haystack

print(f"   • Indexing {len(unique_contexts)} docs...")
docs = [Document(page_content=text) for text in unique_contexts]
vector_db = Chroma.from_documents(docs, hf_embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 20}) 

# Select 20 Random Questions
NUM_SAMPLES = 10
test_data = squad.shuffle(seed=42).select(range(NUM_SAMPLES))
print(f"   • Selected {NUM_SAMPLES} questions.")

# 3. DEFINE METHODS
def method_1_baseline(query):
    return vector_db.as_retriever(search_kwargs={"k": 5}).invoke(query)

def method_2_weighted_retrieval(query, initial_k=25, final_k=5): 
    # Standard Weighted Logic
    candidates = vector_db.similarity_search(query, k=initial_k)
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    
    final_docs = []
    for score, doc in scored_docs[:final_k]:
        doc.metadata['relevance_score'] = float(score)
        final_docs.append(doc)
    return final_docs

def method_4_reordered(query):
    docs = method_2_weighted_retrieval(query, final_k=5)
    return reordering.transform_documents(docs)

def method_5_adversarial(query, initial_k=25, final_k=5):
    # ⚠️ THE RESEARCH METHOD: "The Liar"
    candidates = vector_db.similarity_search(query, k=initial_k)
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    
    # Take the Top-K best docs
    top_docs = scored_docs[:final_k]
    
    # 😈 THE ATTACK: Invert the scores
    # Get the actual scores list [0.99, 0.85, 0.50...]
    actual_scores = [s for s, d in top_docs]
    # Sort them Low-to-High [0.50, 0.85, 0.99]
    inverted_scores = sorted(actual_scores) 
    
    final_docs = []
    # Assign the WORST score to the BEST doc
    for i, (original_score, doc) in enumerate(top_docs):
        fake_score = inverted_scores[i] # top_docs[0] gets lowest score
        doc.metadata['relevance_score'] = float(fake_score)
        final_docs.append(doc)
        
    return final_docs

# Define 5 Methods
methods = {
    "Baseline": method_1_baseline,
    "Weighted (Hidden)": method_2_weighted_retrieval,
    "Weighted (Score-Aware)": method_2_weighted_retrieval, # Logic same, Prompt differs
    "Reordered": method_4_reordered,
    "Adversarial (The Liar)": method_5_adversarial # Logic differs (Inverted scores)
}

# 4. GENERATION LOOP
data = {'question': [], 'answer': [], 'contexts': [], 'ground_truth': [], 'method': []}

print(f"\n🧪 STARTING RESEARCH GENERATION...")
for i, row in enumerate(test_data):
    question = row['question']
    ground_truth = row['answers']['text'][0] 
    print(f"   [{i+1}/{NUM_SAMPLES}] Q: {question[:40]}...")

    for method_name, func in methods.items():
        retrieved_docs = func(question)
        
        # --- PROMPT STRATEGY ---
        context_parts = []
        
        # Group 1: Methods that SHOW scores (The "Smart" ones + The "Liar")
        if method_name in ["Weighted (Score-Aware)", "Adversarial (The Liar)"]:
            for doc in retrieved_docs:
                score = doc.metadata.get('relevance_score', 0.0)
                part = f"[Confidence: {score:.2f}] {doc.page_content}"
                context_parts.append(part)
            instruction = "Trust [Confidence: High] docs. Ignore low confidence noise."
            
        # Group 2: Methods that HIDE scores (Baseline, Hidden, Reordered)
        else:
            for doc in retrieved_docs:
                part = f"- {doc.page_content}"
                context_parts.append(part)
            instruction = "Use the context below."

        context_text = "\n".join(context_parts)
        
        prompt = f"""
        Answer based ONLY on context. {instruction}
        Context:
        {context_text}
        Question: {question}
        """
        
        response = generator_llm.invoke(prompt).content
        
        data['question'].append(question)
        data['answer'].append(response)
        # Pass CLEAN content to Ragas (Hide the lie from the Judge)
        data['contexts'].append([d.page_content for d in retrieved_docs])
        data['ground_truth'].append(ground_truth)
        data['method'].append(method_name)

# 💾 CHECKPOINT
df_gen = pd.DataFrame(data)
df_gen.to_csv("squad_research_checkpoint.csv", index=False)
print("✅ Checkpoint saved: 'squad_research_checkpoint.csv'")

# 5. EVALUATION
print("\n⚖️  STARTING JUDGE (May take 25-30 mins)...")
ragas_features = Features({
    'question': Value('string'),
    'answer': Value('string'),
    'contexts': Sequence(Value('string')),
    'ground_truth': Value('string'),
    'method': Value('string')
})
dataset = Dataset.from_dict(data, features=ragas_features)

# Higher timeout for 5 methods
custom_run_config = RunConfig(max_workers=1, timeout=1500)

try:
    results = evaluate(
        dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
        llm=judge_llm, 
        embeddings=hf_embeddings,
        run_config=custom_run_config
    )
    
    df_scores = results.to_pandas()
    df_scores['method'] = data['method']
    
    # 💾 FINAL SAVE
    df_scores.to_csv("squad_research_results.csv", index=False)
    print("✅ Success! Scores saved to: 'squad_research_results.csv'")

except Exception as e:
    print(f"\n❌ Evaluation Crashed: {e}")
    print("👉 Use 'squad_research_checkpoint.csv' to recover your data.")