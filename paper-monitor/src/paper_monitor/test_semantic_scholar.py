import os
import requests

API_KEY=os.getenv("SEMANTIC_SCHOLAR_API_KEY")

def test_semantic_scholar_key():
    if not API_KEY:
        raise RuntimeError("Missing SEMANTIC_SCHOLAR_API_KEY")

    url="https://api.semanticscholar.org/graph/v1/paper/search"
    params={
        "query":"DeepSeek R1",
        "limit":1,
        "fields":"title,year,citationCount,influentialCitationCount,tldr,externalIds"
    }
    headers={
        "x-api-key":API_KEY
    }

    response=requests.get(url,params=params,headers=headers,timeout=20)
    response.raise_for_status()
    data=response.json()

    print(data)

if __name__=="__main__":
    test_semantic_scholar_key()