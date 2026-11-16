import time
import streamlit as st
import pandas as pd
import plotly.express as px
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(page_title="Global Personal Finance Advisor", page_icon="💰", layout="wide")
st.title("💰 Global Personal Finance Advisor")
st.caption("RAG-based financial education and illustrative planning tools")
countries={"All (exploratory)":"ALL","Global-only":"GLOBAL","United States":"US","India":"IN","United Kingdom":"GB","Canada":"CA","Australia":"AU"}
country=countries[st.sidebar.selectbox("Jurisdiction",list(countries))]
currency=st.sidebar.text_input("Currency code","USD").upper()
api_key = st.secrets.get("GEMINI_API_KEY", "")
model=st.sidebar.text_input("Gemini model","gemini-3.8-flash")
page=st.sidebar.radio("Modules",["Finance Chat","Budget","Investment","Loans","Tax","Insurance","Retirement","Financial Goals"])

@st.cache_resource
def load_db():
    base=Path(__file__).parent
    embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    db=chromadb.PersistentClient(path=str(base/"chroma_db"))
    return embedder,db.get_collection("global_finance_knowledge")
try:
    embedder,col=load_db()
except Exception as e:
    st.error("Global ChromaDB unavailable. Package the database folder."); st.exception(e); st.stop()

def retrieve(q,category=None,k=5):
    if col.count()==0: return []
    qv=embedder.encode([q],normalize_embeddings=True).tolist()
    fs=[]
    if country=="GLOBAL": fs.append({"country":"GLOBAL"})
    elif country!="ALL": fs.append({"$or":[{"country":country},{"country":"GLOBAL"}]})
    if category: fs.append({"category":category})
    where=fs[0] if len(fs)==1 else ({"$and":fs} if len(fs)>1 else None)
    args={"query_embeddings":qv,"n_results":min(k,col.count()),"include":["documents","metadatas"]}
    if where: args["where"]=where
    result=col.query(**args)
    return [{"text":d,**m} for d,m in zip(result["documents"][0],result["metadatas"][0])]


def rag(q, category=None):
    sensitive = [
        "tax rate", "tax deduction", "tax filing",
        "tax bracket", "tax law", "regulation",
        "tax liability"
    ]

    if country == "ALL" and any(x in q.lower() for x in sensitive):
        return "Select a country for jurisdiction-specific information.", []

    docs = retrieve(q, category)

    if not docs:
        return "No relevant evidence found for this jurisdiction/topic.", []

    if not api_key:
        return "Gemini API key is missing. Please configure it in Streamlit Secrets.", []

    # Build context from retrieved documents
    context = "\n\n".join([
        f"Source: {d.get('source')} | "
        f"Publisher: {d.get('publisher')} | "
        f"Country: {d.get('country')} | "
        f"Page: {d.get('page')}\n{d['text']}"
        for d in docs
    ])

    prompt = f"""
You are an educational global finance assistant.

Instructions:
- Do not mix financial rules from different countries.
- Do not invent laws, tax rates, interest rates or investment returns.
- Clearly disclose when evidence is missing.
- Do not provide personalized buy/sell recommendations.
- Use the supplied documents as your primary evidence.
- Cite document filenames and page numbers where available.

Retrieved context:
{context}

User question:
{q}
"""

    refs = [
        {
            "source": d.get("source"),
            "publisher": d.get("publisher"),
            "country": d.get("country"),
            "page": d.get("page")
        }
        for d in docs
    ]

   
    try:
        with genai.Client(api_key=api_key) as client:
            response = None

            # Retry temporary Gemini server overloads
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={"temperature": 0.2}
                    )
                    break

                except Exception as e:
                    error_text = str(e)

                    is_overloaded = (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                    )

                    if not is_overloaded or attempt == 2:
                        raise

                    wait_seconds = 3 * (2 ** attempt)
                    st.warning(
                        f"Gemini is busy. Retrying in {wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)

        answer = response.text or "Gemini returned an empty response."
        return answer, refs

    except Exception as e:
        error_text = str(e)

        if "503" in error_text or "UNAVAILABLE" in error_text:
            return (
                "Gemini is temporarily overloaded. "
                "Please try again in a few minutes.",
                refs
            )

        st.error(f"Gemini API error: {type(e).__name__}: {e}")
        return "Unable to generate an answer. Please try again.", refs



def emi_calc(p,rate,months):
    if p<=0 or months<=0 or rate<0: raise ValueError("Invalid loan inputs")
    r=rate/1200
    e=p/months if r==0 else p*r*(1+r)**months/((1+r)**months-1)
    return e,e*months,e*months-p
def fv(monthly,rate,years):
    n=int(years*12); r=rate/1200
    return monthly*n if r==0 else monthly*((1+r)**n-1)/r*(1+r)

if page=="Finance Chat":
    st.header("AI Finance Chat"); q=st.chat_input("Ask a financial education question")
    if q:
        with st.spinner("Retrieving and generating..."): ans,refs=rag(q)
        st.markdown(ans)
        if refs: st.dataframe(pd.DataFrame(refs))
elif page=="Budget":
    st.header("Monthly Budget Planner"); income=st.number_input("Monthly income",min_value=0.0,value=4000.0)
    cats=["Housing","Food","Transport","Utilities","Healthcare","Education","Shopping","Other"]
    expenses={c:st.number_input(c,min_value=0.0,value=0.0,key="b_"+c) for c in cats}
    total=sum(expenses.values()); balance=income-total
    a,b,c=st.columns(3); a.metric("Income",f"{currency} {income:,.2f}"); b.metric("Expenses",f"{currency} {total:,.2f}"); c.metric("Remaining",f"{currency} {balance:,.2f}")
    st.metric("Remaining %",f"{balance/income*100 if income else 0:.2f}%")
    if total: st.plotly_chart(px.pie(names=list(expenses),values=list(expenses.values())),use_container_width=True)
elif page=="Investment":
    st.header("Illustrative Investment Projection"); st.warning("Assumed returns are hypothetical, not guaranteed.")
    m=st.number_input("Monthly contribution",min_value=0.0,value=300.0); r=st.number_input("Assumed annual return (%)",min_value=0.0,value=7.0); y=st.number_input("Years",min_value=1,value=10)
    if st.button("Calculate"):
        invested=m*y*12; projected=fv(m,r,y); st.metric("Contributed",f"{currency} {invested:,.2f}"); st.metric("Illustrative value",f"{currency} {projected:,.2f}")
    q=st.text_input("Ask investment education question")
    if st.button("Explain investment") and q:
        ans,refs=rag(q,"investment"); st.markdown(ans)
        if refs: st.dataframe(pd.DataFrame(refs))
elif page=="Loans":
    st.header("Loan EMI Calculator"); p=st.number_input("Principal",min_value=1.0,value=200000.0); r=st.number_input("Annual rate (%)",min_value=0.0,value=8.0); n=st.number_input("Months",min_value=1,value=60)
    if st.button("Calculate EMI"):
        e,t,i=emi_calc(p,r,int(n)); a,b,c=st.columns(3); a.metric("Monthly EMI",f"{currency} {e:,.2f}"); b.metric("Total payment",f"{currency} {t:,.2f}"); c.metric("Interest",f"{currency} {i:,.2f}")
elif page=="Tax":
    st.header("Tax Education"); st.warning("Tax depends on tax residence and tax year; this is not a tax calculator.")
    q=st.text_area("Tax question")
    if country in ("ALL","GLOBAL"): st.info("Select a specific country for tax questions.")
    if st.button("Search tax knowledge") and q and country not in ("ALL","GLOBAL"):
        ans,refs=rag(q,"taxation"); st.markdown(ans)
        if refs: st.dataframe(pd.DataFrame(refs))
elif page=="Insurance":
    st.header("Insurance Coverage Gap"); existing=st.number_input("Existing cover",min_value=0.0,value=100000.0); target=st.number_input("Your selected target",min_value=0.0,value=250000.0)
    st.metric("Difference",f"{currency} {max(0,target-existing):,.2f}"); st.caption("Not a recommended coverage amount.")
    q=st.text_input("Insurance question")
    if st.button("Explain insurance") and q:
        ans,refs=rag(q,"insurance"); st.markdown(ans)
        if refs: st.dataframe(pd.DataFrame(refs))
elif page=="Retirement":
    st.header("Retirement Projection"); corpus=st.number_input("Current corpus",min_value=0.0,value=10000.0); monthly=st.number_input("Monthly contribution",min_value=0.0,value=400.0); rate=st.number_input("Assumed annual return (%)",min_value=0.0,value=6.0); years=st.number_input("Years",min_value=1,value=20)
    if st.button("Calculate retirement"):
        value=corpus*(1+rate/1200)**(years*12)+fv(monthly,rate,years); st.metric("Illustrative corpus",f"{currency} {value:,.2f}"); st.caption("Excludes inflation, taxes, fees and changing returns.")
elif page=="Financial Goals":
    st.header("Financial Goal Planner"); target=st.number_input("Target amount",min_value=0.0,value=50000.0); current=st.number_input("Current savings",min_value=0.0,value=10000.0); years=st.number_input("Years",min_value=1,value=3); rate=st.number_input("Assumed annual return (%)",min_value=0.0,value=0.0)
    if st.button("Calculate monthly target"):
        rem=max(0,target-current); n=years*12; r=rate/1200; monthly=rem/n if r==0 else rem*r/((1+r)**n-1)/(1+r); st.metric("Illustrative monthly saving",f"{currency} {monthly:,.2f}")
st.divider(); st.caption("Educational prototype. Verify current rules with official sources and qualified professionals.")
