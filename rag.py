"""
Agricultural RAG System - Complete Implementation
Single file: rag.py
Run with: python rag.py
Access at: http://localhost:8000
"""

import gradio as gr
from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langserve import add_routes
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
from typing import List, Dict, Any
import json
from datetime import datetime
import os

# Download NLTK data for evaluation
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("NLTK downloads failed but continuing...")

# ============================================================================
# 1. EXPANDED AGRICULTURAL KNOWLEDGE BASE
# ============================================================================

AGRICULTURAL_DOCS = [
    # ===== CROPS =====
    # Maize/Corn
    "Maize (Zea mays), also known as corn, is a cereal grain first domesticated in Mexico. It requires 120-140 days to mature, needs 500-800mm of rainfall, and grows best in temperatures 18-32°C.",
    "Corn production worldwide exceeds 1.1 billion metric tons annually. The United States, China, and Brazil are the top producers. Maize is used for human consumption, animal feed, and biofuel production.",
    "Sweet corn is harvested immature and eaten as a vegetable. Field corn is harvested mature for grain. Dent corn, flint corn, and popcorn are major varieties with different endosperm types.",
    "Corn pests include European corn borer, corn rootworm, and fall armyworm. Diseases like corn smut, gray leaf spot, and northern corn leaf blight can significantly reduce yields.",
    
    # Wheat
    "Wheat (Triticum spp.) is a grass cultivated for its grain. It's the second most produced cereal after maize. Winter wheat is planted in fall, spring wheat in spring. Optimal growing temperature is 15-20°C.",
    "Wheat requires well-drained loamy soil with pH 6.0-7.5. It needs 300-500mm of water during growing season. Major producers: China, India, Russia, and the United States.",
    "Wheat diseases include rust (stem, leaf, stripe), Fusarium head blight, and powdery mildew. Pest threats include Hessian fly, wheat stem sawfly, and aphids.",
    "Wheat grain is processed into flour for bread, pasta, and pastries. Whole wheat contains bran, germ, and endosperm; white flour has bran and germ removed.",
    
    # Rice
    "Rice (Oryza sativa) is a staple food for over half the world's population. It requires high temperatures (20-40°C), high humidity, and standing water during growth. Paddy fields need 1000-2000mm water.",
    "Rice cultivation methods include transplanting seedlings into flooded paddies or direct seeding. Lowland rice is flooded, upland rice depends on rainfall. Major producers: China, India, Indonesia, Bangladesh.",
    "Brown rice is whole grain with only husk removed. White rice is milled to remove bran. Parboiled rice is steamed before milling to retain nutrients. Aromatic varieties include basmati and jasmine.",
    "Rice pests: stem borers, brown planthoppers, and gall midge. Diseases: blast, bacterial leaf blight, and sheath blight. Integrated pest management reduces chemical use.",
    
    # Soybeans
    "Soybean (Glycine max) is a legume rich in protein and oil. It fixes atmospheric nitrogen through symbiosis with Bradyrhizobium bacteria, enriching soil. Requires 20-30°C temperature and 450-700mm rainfall.",
    "Soybeans are processed for oil (18-20% of seed weight) and meal (used for animal feed). Whole soybeans are used for tofu, tempeh, soy milk, and edamame. Major producers: USA, Brazil, Argentina.",
    "Soybean cyst nematode is a major pest. Diseases include soybean rust, frogeye leaf spot, and sudden death syndrome. Crop rotation with corn helps break pest cycles.",
    
    # Potatoes
    "Potato (Solanum tuberosum) is the world's fourth largest food crop. It's a cool-season crop grown from tubers, not true seeds. Optimal temperature 15-20°C; frost kills foliage. Requires well-drained, loose soil.",
    "Potato varieties include russet (baking), red (boiling), yellow (buttery flavor), and purple (antioxidant-rich). Potatoes are harvested 90-120 days after planting depending on variety.",
    "Late blight (Phytophthora infestans) caused the Irish Potato Famine. Other diseases: early blight, potato virus Y, and bacterial ring rot. Pests: Colorado potato beetle, aphids, and wireworms.",
    
    # ===== LIVESTOCK =====
    "Cattle farming includes beef production (for meat) and dairy (for milk). Beef breeds: Angus, Hereford. Dairy breeds: Holstein, Jersey. Cattle need pasture or feed, water, minerals, and veterinary care.",
    "Dairy cows are milked 2-3 times daily. Average Holstein produces 22-30 liters/day. Lactation lasts 305 days followed by 60-day dry period. Mastitis is common udder infection requiring treatment.",
    "Poultry farming: broilers for meat (6-8 weeks to market), layers for eggs (start laying at 18-20 weeks). Chickens need balanced feed, clean water, ventilation, and protection from predators.",
    "Swine production: sows farrow (give birth) to 10-12 piglets twice yearly. Pigs are raised to market weight (110-120kg) in 5-6 months. Common diseases: porcine reproductive and respiratory syndrome (PRRS), swine flu.",
    
    # ===== SOIL SCIENCE =====
    "Soil texture (sand, silt, clay proportions) affects water holding capacity and drainage. Loam (40% sand, 40% silt, 20% clay) is ideal for most crops. Clay soils hold nutrients but drain poorly.",
    "Soil pH affects nutrient availability. Most crops prefer pH 6.0-7.0 (slightly acidic to neutral). Acidic soils (pH <5.5) need lime application. Alkaline soils (pH >8) may need sulfur or organic matter.",
    "Soil organic matter (SOM) improves structure, water retention, and nutrient supply. Aim for 3-5% SOM. Sources: compost, cover crops, manure, crop residues. Tillage accelerates SOM loss.",
    "Nitrogen (N) promotes leaf growth. Phosphorus (P) supports root development and flowering. Potassium (K) improves disease resistance and water regulation. NPK ratios in fertilizers match crop needs.",
    "Soil testing every 2-3 years guides fertilizer application. Samples should be taken from multiple locations, mixed, and sent to lab. Tests measure pH, organic matter, N, P, K, and micronutrients.",
    
    # ===== IRRIGATION & WATER MANAGEMENT =====
    "Drip irrigation delivers water directly to plant roots through tubes and emitters. Efficiency 90-95% (vs 50-70% for sprinklers). Reduces evaporation, runoff, and weed growth. Ideal for row crops and orchards.",
    "Sprinkler irrigation sprays water like rain. Types: center pivot (circular fields), lateral move (rectangular), and solid set (permanent). Efficiency 70-85%. Can be used for frost protection and cooling.",
    "Surface irrigation (furrow, basin, border) uses gravity to distribute water. Oldest method, low energy cost but lower efficiency (40-60%). Requires land leveling and can cause runoff and erosion.",
    "Deficit irrigation applies less water than full crop requirement during drought-tolerant growth stages. Saves water with minimal yield reduction. Requires careful monitoring of soil moisture.",
    "Rainwater harvesting collects runoff from roofs or land into ponds or tanks. Provides supplemental irrigation. One cubic meter of storage per 10 square meters of catchment area is typical.",
    
    # ===== PESTS & DISEASES =====
    "Aphids are small sap-sucking insects that transmit plant viruses. Natural enemies: ladybugs, lacewings, parasitic wasps. Control: insecticidal soaps, neem oil, or resistant varieties.",
    "Corn borers tunnel into stalks, causing lodging and yield loss. European corn borer is major pest. Control: Bt corn (genetically modified for resistance), biological control with Trichogramma wasps.",
    "Fall armyworm is devastating pest of maize in Africa and Asia. Larvae feed on leaves and ears. Control: early planting, pheromone traps, biopesticides (neem, Bacillus thuringiensis).",
    "Integrated Pest Management (IPM) combines biological, cultural, physical, and chemical controls. Steps: monitoring, identification, action thresholds, prevention, and multiple control tactics.",
    "Plant diseases caused by fungi (powdery mildew, rusts), bacteria (leaf spot, wilt), viruses (mosaic, leaf curl), and nematodes (root knot). Prevention: resistant varieties, crop rotation, sanitation.",
    
    # ===== ORGANIC & SUSTAINABLE FARMING =====
    "Organic farming prohibits synthetic pesticides, fertilizers, GMOs, and antibiotics. Uses compost, green manure, crop rotation, biological pest control. Certification requires 3-year transition period.",
    "Cover crops (clover, rye, vetch) planted between cash crops prevent erosion, suppress weeds, fix nitrogen, and improve soil structure. Terminated by mowing, rolling, or herbicides before planting main crop.",
    "Crop rotation sequences: corn (grass family) followed by soybeans (legume) followed by wheat (cereal) gives diversity. Breaks pest cycles, improves soil fertility, and reduces need for inputs.",
    "Conservation tillage (no-till, reduced till) leaves crop residues on surface. Reduces erosion, builds soil organic matter, saves fuel. May require more herbicides for weed control.",
    "Agroforestry integrates trees with crops (alley cropping) or livestock (silvopasture). Trees provide shade, windbreaks, timber, fruit, and improve soil through leaf litter and nitrogen fixation.",
    
    # ===== PRECISION AGRICULTURE =====
    "Precision agriculture uses GPS, sensors, drones, and variable rate technology to optimize inputs. Yield maps show field variability. Soil maps guide variable fertilizer application.",
    "GPS-guided tractors enable auto-steering, reducing overlap and operator fatigue. Accuracy within 2-3cm with RTK correction. Saves fuel, seed, fertilizer, and time.",
    "Drones with multispectral cameras monitor crop health through NDVI (Normalized Difference Vegetation Index). Detects stress from water, nutrients, pests before visible to naked eye.",
    "Variable rate technology (VRT) applies seeds, fertilizer, or chemicals at different rates across field based on soil maps or real-time sensors. Increases efficiency and reduces environmental impact.",
    
    # ===== AGRICULTURAL ECONOMICS =====
    "Commodity prices fluctuate with supply, demand, weather, and global trade. Futures markets help farmers lock in prices. Basis is difference between local cash price and futures price.",
    "Farm subsidies include direct payments, crop insurance subsidies, and disaster assistance. Programs vary by country. In US, Farm Bill sets policy every 5 years.",
    "Crop insurance protects against yield loss (from weather, pests) or revenue loss (price declines). Premiums partially subsidized by government. Indemnity payments triggered when yield or revenue falls below guarantee.",
    "Supply and demand fundamentals: global population growth (9.7 billion by 2050) drives food demand. Income growth in developing countries shifts diets toward more meat, dairy, and processed foods.",
]

# ============================================================================
# 2. EVALUATION DATASET
# ============================================================================

EVALUATION_QUESTIONS = [
    "What are the ideal growing conditions for wheat?",
    "How does drip irrigation conserve water?",
    "Name two common pests in corn farming and how to control them.",
    "What is the role of nitrogen in plant growth?",
    "Explain crop rotation and its benefits.",
    "What is organic farming and what does certification require?",
    "How does soil pH affect crop growth and how can it be adjusted?",
    "What is integrated pest management (IPM)?",
    "Compare different irrigation methods and their efficiencies.",
    "What is precision agriculture and what technologies does it use?",
    "How do cover crops improve soil health?",
    "What are the main differences between beef and dairy cattle farming?",
    "Describe the rice cultivation process and water requirements.",
    "What causes late blight in potatoes and why is it historically significant?",
    "How do commodity futures markets help farmers manage risk?",
]

GROUND_TRUTH_ANSWERS = [
    "Wheat grows best in well-drained loamy soil with pH 6.0-7.5, temperatures of 15-20°C, and 300-500mm of water during the growing season. Winter wheat is planted in fall, spring wheat in spring.",
    "Drip irrigation conserves water by delivering it directly to plant roots through tubes and emitters, achieving 90-95% efficiency. It reduces evaporation, runoff, and weed growth compared to sprinklers (70-85%) or surface irrigation (40-60%).",
    "Two common corn pests are: 1) European corn borer - tunnels into stalks; control with Bt corn or Trichogramma wasps. 2) Fall armyworm - feeds on leaves and ears; control with early planting, pheromone traps, or biopesticides like neem and Bacillus thuringiensis.",
    "Nitrogen is a primary macronutrient essential for leaf growth and chlorophyll production. It's a key component of amino acids, proteins, and chlorophyll. Nitrogen deficiency causes yellowing (chlorosis) and stunted growth.",
    "Crop rotation is planting different crops sequentially on the same land to improve soil fertility, break pest cycles, and reduce disease pressure. For example, corn (grass family) followed by soybeans (legume) adds nitrogen, followed by wheat provides diversity.",
    "Organic farming prohibits synthetic pesticides, fertilizers, GMOs, and antibiotics. It uses compost, green manure, crop rotation, and biological pest control. Certification requires a 3-year transition period with documented practices and annual inspections.",
    "Soil pH affects nutrient availability - most crops prefer pH 6.0-7.0. Acidic soils (pH <5.5) need lime application to raise pH. Alkaline soils (pH >8) may need sulfur or organic matter to lower pH. Soil testing guides amendments.",
    "Integrated Pest Management (IPM) combines biological, cultural, physical, and chemical controls. Steps include monitoring pests, correct identification, establishing action thresholds, prevention through practices like crop rotation, and using multiple control tactics when thresholds are exceeded.",
    "Drip irrigation: 90-95% efficiency, delivers water to roots, minimal evaporation. Sprinkler: 70-85%, mimics rain, can be used for frost protection. Surface irrigation: 40-60%, uses gravity, low energy but higher water loss. Choice depends on crop, climate, and economics.",
    "Precision agriculture uses GPS, sensors, drones, and variable rate technology to optimize inputs. GPS-guided tractors enable auto-steering. Drones with multispectral cameras monitor crop health through NDVI. Variable rate technology applies inputs at different rates across fields based on soil maps.",
    "Cover crops (clover, rye, vetch) planted between cash crops prevent erosion, suppress weeds, fix nitrogen (legumes), and improve soil structure with their root systems. When terminated, they add organic matter as green manure.",
    "Beef cattle farming raises animals for meat using breeds like Angus and Hereford. Dairy farming produces milk using breeds like Holstein and Jersey; cows are milked 2-3 times daily and produce 22-30 liters/day during 305-day lactation. Both need pasture/feed, water, minerals, and veterinary care.",
    "Rice requires high temperatures (20-40°C), high humidity, and standing water during growth. Paddy fields need 1000-2000mm water. Methods include transplanting seedlings into flooded paddies or direct seeding. Lowland rice is flooded; upland rice depends on rainfall.",
    "Late blight (Phytophthora infestans) is a fungal-like disease causing rapid decay of potato foliage and tubers. It caused the Irish Potato Famine (1845-1852), leading to mass starvation and emigration. Control requires resistant varieties, fungicides, and destroying infected plants.",
    "Commodity futures markets allow farmers to lock in prices for future delivery, reducing price risk. Farmers can sell futures contracts to guarantee a price. Basis (difference between local cash price and futures) affects actual returns. Crop insurance complements futures by protecting against yield loss.",
]

# ============================================================================
# 3. RAG SYSTEM CLASS
# ============================================================================

class AgriculturalRAGSystem:
    """Complete RAG system for agricultural Q&A with evaluation capabilities."""
    
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        self.setup_system()
    
    def setup_system(self):
        """Initialize all components of the RAG system."""
        print("🌱 Setting up Agricultural RAG System...")
        
        # 1. Create document chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.create_documents(AGRICULTURAL_DOCS)
        print(f"📄 Created {len(chunks)} document chunks")
        
        # 2. Create embeddings and vector store
        print("🔤 Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = FAISS.from_documents(chunks, embedding_model)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        print("✅ Vector store created")
        
        # 3. Load local LLM
        print(f"🤖 Loading LLM: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.3,
            device=-1  # CPU
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # 4. Create prompt template
        prompt_template = """You are an expert agricultural assistant. Use the following context to answer the question accurately and completely. If the context doesn't contain the answer, say you don't know - don't make up information.

Context:
{context}

Question: {question}

Provide a clear, helpful answer based only on the context above:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 5. Create RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=False
        )
        print("✅ RAG chain ready\n")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a single query and return answer with sources."""
        result = self.qa_chain({"query": question})
        return {
            "question": question,
            "answer": result["result"],
            "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
        }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries."""
        return [self.query(q) for q in questions]

# ============================================================================
# 4. EVALUATION METRICS
# ============================================================================

class RAGEvaluator:
    """Evaluate RAG system performance using various metrics."""
    
    def __init__(self, rag_system: AgriculturalRAGSystem):
        self.rag_system = rag_system
        self.smoothing = SmoothingFunction().method1
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate answers."""
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        if len(candidate_tokens) == 0:
            return 0.0
        
        return sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=self.smoothing
        )
    
    def calculate_meteor(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score (requires wordnet)."""
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            return meteor_score([reference_tokens], candidate_tokens)
        except:
            # Fallback if meteor fails
            return self.calculate_bleu(reference, candidate)
    
    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        """Simple relevance metric - checks if answer contains key terms from question."""
        question_words = set(nltk.word_tokenize(question.lower()))
        answer_words = set(nltk.word_tokenize(answer.lower()))
        
        common_words = question_words.intersection(answer_words)
        
        # Remove common stopwords
        stopwords = {'what', 'how', 'why', 'when', 'where', 'who', 'which',
                     'is', 'are', 'was', 'were', 'do', 'does', 'did',
                     'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at'}
        common_words = common_words - stopwords
        question_words = question_words - stopwords
        
        if len(question_words) == 0:
            return 0.5
        
        return len(common_words) / len(question_words)
    
    def evaluate(self, questions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """Run full evaluation on test set."""
        print("🔍 Running evaluation...")
        
        results = []
        bleu_scores = []
        meteor_scores = []
        relevance_scores = []
        
        for i, (q, gt) in enumerate(zip(questions, ground_truths)):
            print(f"  Evaluating {i+1}/{len(questions)}: {q[:50]}...")
            
            # Get system answer
            response = self.rag_system.query(q)
            answer = response["answer"]
            
            # Calculate metrics
            bleu = self.calculate_bleu(gt, answer)
            meteor = self.calculate_meteor(gt, answer)
            relevance = self.calculate_answer_relevance(q, answer)
            
            bleu_scores.append(bleu)
            meteor_scores.append(meteor)
            relevance_scores.append(relevance)
            
            results.append({
                "question": q,
                "ground_truth": gt,
                "system_answer": answer,
                "bleu": bleu,
                "meteor": meteor,
                "relevance": relevance,
                "sources": response["sources"]
            })
        
        # Calculate averages
        avg_bleu = np.mean(bleu_scores)
        avg_meteor = np.mean(meteor_scores)
        avg_relevance = np.mean(relevance_scores)
        
        evaluation_summary = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "average_bleu": avg_bleu,
            "average_meteor": avg_meteor,
            "average_relevance": avg_relevance,
            "individual_results": results,
            "score_summary": {
                "bleu": f"{avg_bleu:.3f}",
                "meteor": f"{avg_meteor:.3f}",
                "relevance": f"{avg_relevance:.3f}"
            }
        }
        
        print(f"✅ Evaluation complete!")
        print(f"📊 BLEU: {avg_bleu:.3f}, METEOR: {avg_meteor:.3f}, Relevance: {avg_relevance:.3f}")
        
        return evaluation_summary

def save_evaluation_results(results: Dict[str, Any], filename: str = "evaluation_results.json"):
    """Save evaluation results to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to {filename}")

# ============================================================================
# 5. FASTAPI + LANGSERVE SETUP
# ============================================================================

# Initialize the RAG system
rag_system = AgriculturalRAGSystem()

# Create FastAPI app
app = FastAPI(
    title="🌾 Agricultural RAG Bot",
    description="Ask questions about crops, livestock, soil, irrigation, pests, and sustainable farming",
    version="1.0.0"
)

# Add LangServe route for the chain
add_routes(app, rag_system.qa_chain, path="/chain")

# Add evaluation endpoint
@app.post("/evaluate")
async def run_evaluation():
    """Run evaluation on the test dataset."""
    evaluator = RAGEvaluator(rag_system)
    results = evaluator.evaluate(EVALUATION_QUESTIONS, GROUND_TRUTH_ANSWERS)
    save_evaluation_results(results)
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": rag_system.model_name, "documents": len(AGRICULTURAL_DOCS)}

# ============================================================================
# 6. GRADIO CHAT INTERFACE
# ============================================================================

def format_sources(sources: List[str]) -> str:
    """Format source documents for display."""
    if not sources:
        return "No sources retrieved"
    
    formatted = "\n\n📚 **Sources:**\n"
    for i, source in enumerate(sources, 1):
        formatted += f"{i}. {source}\n"
    return formatted

def chat_with_bot(message, history):
    """Process chat message and return response."""
    try:
        result = rag_system.query(message)
        response = result["answer"]
        
        # Add sources if available
        if result["sources"]:
            response += format_sources(result["sources"])
        
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

def run_evaluation_ui():
    """Run evaluation and display results in Gradio."""
    evaluator = RAGEvaluator(rag_system)
    results = evaluator.evaluate(EVALUATION_QUESTIONS[:5], GROUND_TRUTH_ANSWERS[:5])  # First 5 for speed
    
    output = f"""
    ## 📊 Evaluation Results
    
    **Average BLEU Score:** {results['average_bleu']:.3f}
    **Average METEOR Score:** {results['average_meteor']:.3f}
    **Average Relevance Score:** {results['average_relevance']:.3f}
    
    ### Sample Results:
    """
    
    for i, res in enumerate(results['individual_results'][:3]):
        output += f"""
    **Q{i+1}:** {res['question']}
    **System:** {res['system_answer'][:150]}...
    **BLEU:** {res['bleu']:.3f}
    ---
    """
    
    return output

# Create Gradio interface
with gr.Blocks(title="🌾 Agricultural Q&A System", theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown("""
    # 🌾 Agricultural Knowledge Assistant
    
    Ask questions about:
    - 🌽 Crops (corn, wheat, rice, soybeans, potatoes)
    - 🐄 Livestock (cattle, poultry, swine)
    - 🌱 Soil science and fertilizers
    - 💧 Irrigation and water management
    - 🐛 Pests and diseases
    - 🌿 Organic and sustainable farming
    - 📊 Precision agriculture
    - 💰 Agricultural economics
    """)
    
    with gr.Tab("💬 Chat"):
        chatbot = gr.ChatInterface(
            fn=chat_with_bot,
            title="Agricultural Expert",
            description="Ask anything about farming and agriculture",
            theme="soft",
        )
    
    with gr.Tab("🔍 Evaluation"):
        gr.Markdown("## RAG System Evaluation")
        gr.Markdown("Run evaluation on test questions to measure system performance.")
        
        eval_btn = gr.Button("Run Evaluation", variant="primary")
        eval_output = gr.Markdown()
        
        eval_btn.click(
            fn=run_evaluation_ui,
            outputs=eval_output
        )
    
    with gr.Tab("📚 Knowledge Base"):
        gr.Markdown(f"## Knowledge Base Statistics")
        gr.Markdown(f"- **Total documents:** {len(AGRICULTURAL_DOCS)}")
        gr.Markdown(f"- **Test questions:** {len(EVALUATION_QUESTIONS)}")
        gr.Markdown(f"- **Model:** {rag_system.model_name}")
        
        sample_docs = "\n\n".join(AGRICULTURAL_DOCS[:5])
        gr.Markdown(f"### Sample Documents:\n{sample_docs}")
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About This System
        
        This is a **Retrieval-Augmented Generation (RAG)** system for agricultural knowledge.
        
        ### Components:
        - **Retriever:** FAISS vector store with sentence-transformers embeddings
        - **Generator:** Google Flan-T5-small language model
        - **Interface:** Gradio chat UI + FastAPI/LangServe
        
        ### Features:
        - 📖 50+ agricultural documents
        - 🔍 Semantic search for relevant context
        - 🤖 Local LLM (no API keys needed)
        - 📊 Built-in evaluation metrics
        - 🌐 REST API via LangServe
        
        ### Evaluation Metrics:
        - **BLEU:** Measures n-gram overlap with ground truth
        - **METEOR:** Considers synonyms and stemming
        - **Relevance:** Keyword overlap between question and answer
        
        ### Deployment:
        - Run locally: `python rag.py`
        - Access: http://localhost:8000
        - API docs: http://localhost:8000/docs
        """)

# Mount Gradio app on FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")

# ============================================================================
# 7. MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    from threading import Timer
    
    def open_browser():
        webbrowser.open("http://localhost:8000")
    
    print("=" * 60)
    print("🌾 AGRICULTURAL RAG SYSTEM")
    print("=" * 60)
    print(f"📚 Knowledge base: {len(AGRICULTURAL_DOCS)} documents")
    print(f"🤖 Model: {rag_system.model_name}")
    print(f"🧪 Test questions: {len(EVALUATION_QUESTIONS)}")
    print("=" * 60)
    print("🚀 Starting server...")
    print("📱 Open http://localhost:8000 in your browser")
    print("📚 API docs at http://localhost:8000/docs")
    print("🔍 Evaluation endpoint: POST /evaluate")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    
    # Open browser automatically after 2 seconds
    Timer(2, open_browser).start()
    
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)