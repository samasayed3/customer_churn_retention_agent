"""
Compare chunk sizes on a small question set.

    python -m vector_db.eval_chunks

For each (chunk_size, overlap) pair it builds a temporary in-memory index
(your real chroma_store is NOT touched), asks every question, and checks
whether the expected phrase appears in the top-1 / top-3 retrieved chunks.
"""
import chromadb
from sentence_transformers import SentenceTransformer

from config import POLICIES_DIR
from vector_db.build_vectordb import split_text
from vector_db.settings import EMBEDDING_MODEL

# (question, phrase that MUST appear in the right chunk)
# Add your own questions here -- 15-20 is a good size.
TESTS = [
    ('What is the penalty if I leave a 12-month contract early?',
     '100 USD'),
    ('What happens to my contract if I do nothing when my One year term ends?',
     'moves to a Month-to-month contract on the same plan'),
    ('Can I switch to a shorter contract in the middle of my term?',
     'only at renewal'),
    ('Can a Phone Only customer take a 24-month contract?',
     'Phone Only plan is not available with the Two year contract'),
    ('I am on a monthly contract and want the Premium Bundle. What do I have to do?',
     'must choose a One year or Two year contract'),
    ('How many days do I have to send back the modem after I cancel?',
     'within 14 days'),
    ('How much notice do I need to give before cancelling?',
     'notice period of 30 days for every contract type'),
    ('What do I lose if I cancel while my Loyalty Reward is running?',
     'loses the remaining benefit of that offer'),
    ('How much is the late payment charge?',
     'late fee of 8 USD'),
    ('How long do I have to complain about a wrong charge?',
     'within 60 days of the bill date'),
    ('Can I turn my leftover bill credit into money?',
     'no cash value'),
    ('Which offer gives a free streaming month?',
     'one free month of any streaming add-on'),
    ('For how long is my price frozen with the Fiber Price Lock?',
     'freezes the monthly price for 24 months'),
    ('Which tenure range makes someone a Growing Customer?',
     'tenure of 13 to 24 months'),
    ('At what churn probability is a customer considered high risk?',
     '0.6 or higher is High risk'),
    ('Which plan does not include internet?',
     'has no internet'),
    ('How fast does TeleNova fix an internet outage?',
     'within 48 hours'),
    ('What time does customer support open and close?',
     '8 am to 8 pm'),
    ('Can a Premium Bundle customer get the Free Tech Support Trial?',
     'The Premium Bundle includes tech support and is not in the list of applicable plans'),
    ('If I leave and come back later, is my old tenure kept?',
     'tenure starting again from zero'),
    ('Which customers get a priority support line?',
     'priority support line'),
    ('Where can I change my payment to automatic?',
     'in the TeleNova app, by phone, or in a TeleNova store'),
    ('What will I pay in total if I cancel a Two year contract with 10 months left?',
     'plus the bill for the 30-day notice period'),
    ('Why can a brand new customer not get the Contract Upgrade Discount?',
     'tenure under 6 months'),
]

CONFIGS = [(300, 30), (400, 40), (500, 50), (700, 100)]
TOP_K = 3


def main():
    docs = {f.name: f.read_text(encoding="utf-8")
            for f in sorted(POLICIES_DIR.glob("*.txt"))}
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = model.encode([q for q, _ in TESTS], normalize_embeddings=True).tolist()

    print(f"{'size/overlap':<14}{'chunks':>7}{'avg len':>9}{'hit@1':>8}{'hit@3':>8}")
    for size, overlap in CONFIGS:
        texts = [c for t in docs.values() for c in split_text(t, size, overlap)]
        client = chromadb.EphemeralClient()
        name = f"tmp_{size}_{overlap}"
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        col.add(ids=[str(i) for i in range(len(texts))], documents=texts,
                embeddings=model.encode(texts, normalize_embeddings=True).tolist())

        hit1 = hit3 = 0
        misses = []
        for (q, must), emb in zip(TESTS, q_emb):
            got = col.query(query_embeddings=[emb], n_results=TOP_K)["documents"][0]
            ok1 = must in got[0]
            ok3 = any(must in g for g in got)
            hit1 += ok1
            hit3 += ok3
            if not ok3:
                misses.append(q)
        avg = sum(map(len, texts)) // len(texts)
        n = len(TESTS)
        print(f"{size}/{overlap:<10}{len(texts):>7}{avg:>9}"
              f"{hit1:>5}/{n}{hit3:>5}/{n}")
        for m in misses:
            print(f"    MISSED: {m}")
        client.delete_collection(name)


if __name__ == "__main__":
    main()
