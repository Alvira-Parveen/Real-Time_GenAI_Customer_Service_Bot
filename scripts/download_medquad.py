import os
import requests
import xml.etree.ElementTree as ET
import json

MEDQUAD_DIR = "data/medquad"
RAW_XML_DIR = os.path.join(MEDQUAD_DIR, "raw_xml")
os.makedirs(RAW_XML_DIR, exist_ok=True)

# List of real MedQuAD XML paths from official github repository: abachaa/MedQuAD
SAMPLE_XMLS = [
    ("5_NIDDK_QA", "0000001.xml"), # Acromegaly
    ("5_NIDDK_QA", "0000002.xml"), # Adrenal Insufficiency & Addison's Disease
    ("5_NIDDK_QA", "0000003.xml"), # Amyloidosis and Kidney Disease
    ("5_NIDDK_QA", "0000005.xml"), # Autosomal Dominant Polycystic Kidney Disease
    ("5_NIDDK_QA", "0000008.xml"), # Celiac Disease
    ("5_NIDDK_QA", "0000010.xml"), # Cirrhosis
    ("5_NIDDK_QA", "0000014.xml"), # Crohn's Disease
    ("5_NIDDK_QA", "0000019.xml"), # Diabetes Overview
    ("5_NIDDK_QA", "0000022.xml"), # Diabetic Kidney Disease
    ("5_NIDDK_QA", "0000030.xml"), # Gallstones
    ("9_CDC_QA", "0000002.xml"),   # ADHD
    ("9_CDC_QA", "0000003.xml"),   # Arthritis
    ("9_CDC_QA", "0000004.xml"),   # Asthma
    ("9_CDC_QA", "0000005.xml"),   # Autism Spectrum Disorder
    ("9_CDC_QA", "0000008.xml"),   # Cancer Prevention
    ("9_CDC_QA", "0000012.xml"),   # Diabetes Prevention
    ("9_CDC_QA", "0000015.xml"),   # Epilepsy
    ("9_CDC_QA", "0000020.xml"),   # High Blood Pressure / Hypertension
    ("8_NHLBI_QA_XML", "0000001.xml"), # Angina
    ("8_NHLBI_QA_XML", "0000003.xml"), # Arrhythmia
    ("8_NHLBI_QA_XML", "0000004.xml"), # Atherosclerosis
    ("8_NHLBI_QA_XML", "0000010.xml"), # Coronary Heart Disease
    ("1_CancerGov_QA", "0000001.xml"), # Breast Cancer Screening
    ("1_CancerGov_QA", "0000003.xml"), # Colon Cancer
    ("3_GHR_QA", "0000001.xml"),       # 1p36 deletion syndrome
    ("3_GHR_QA", "0000005.xml"),       # Achondroplasia
]

def download_and_parse():
    print(f"Downloading authentic MedQuAD files from GitHub...")
    parsed_records = []
    
    for folder, filename in SAMPLE_XMLS:
        raw_url = f"https://raw.githubusercontent.com/abachaa/MedQuAD/master/{folder}/{filename}"
        local_xml_path = os.path.join(RAW_XML_DIR, f"{folder}_{filename}")
        
        try:
            resp = requests.get(raw_url, timeout=10)
            if resp.status_code == 200:
                with open(local_xml_path, "wb") as f:
                    f.write(resp.content)
                
                # Parse XML
                root = ET.fromstring(resp.content)
                doc_id = root.attrib.get("id", "")
                source = root.attrib.get("source", folder)
                focus = root.findtext("Focus", default="General Medicine")
                
                # Extract semantic types if present
                semantic_types = []
                for st in root.findall(".//SemanticType"):
                    if st.text:
                        semantic_types.append(st.text.strip())
                
                qa_pairs = root.findall(".//QAPair")
                for qa in qa_pairs:
                    qid = qa.find("Question").attrib.get("qid", "") if qa.find("Question") is not None else ""
                    qtype = qa.find("Question").attrib.get("qtype", "general") if qa.find("Question") is not None else ""
                    question = qa.findtext("Question", default="").strip()
                    answer = qa.findtext("Answer", default="").strip()
                    
                    if question and answer:
                        parsed_records.append({
                            "id": f"{doc_id}_{qid}",
                            "source": source,
                            "folder": folder,
                            "focus": focus,
                            "qtype": qtype,
                            "semantic_types": semantic_types,
                            "question": question,
                            "answer": answer
                        })
                print(f"  Loaded {folder}/{filename} -> {focus} ({len(qa_pairs)} QA pairs)")
            else:
                print(f"  Warning: HTTP {resp.status_code} for {raw_url}")
        except Exception as e:
            print(f"  Error fetching {raw_url}: {e}")
            
    output_json = os.path.join(MEDQUAD_DIR, "medquad_qa.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(parsed_records, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully downloaded and parsed {len(parsed_records)} MedQuAD QA pairs into {output_json}!")
    return len(parsed_records)

if __name__ == "__main__":
    download_and_parse()
