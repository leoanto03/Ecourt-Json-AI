import streamlit as st
import json
import requests
import re
import os # Import the os module
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() # Load environment variables from .env file (if it exists)

# # --- Groq API Configuration ---
# GROQ_API_URL = "https://api.friendli.ai/dedicated/v1/chat/completions"
# GROQ_DEFAULT_MODEL = "c6xp7t1pxbvl"


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama3-8b-8192"
 


# --- Get API Key from Environment Variable ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- !!! DO NOT HARDCODE API KEY HERE ANYMORE !!! ---

# --- Initial check for API Key ---
if not GROQ_API_KEY:
    # This message will show up locally if .env is missing or GROQ_API_KEY isn't in it.
    # On Render, it will show if the environment variable isn't set in the dashboard.
    st.error("FATAL: GROQ_API_KEY environment variable is not set. The application cannot function.")
    st.stop() # Stop the app if the key is missing

# --- Color Codes ---
COLOR_DARK_BLUE = "#122D58"
COLOR_GREEN_RELEVANT = "#28a745"
COLOR_RED_IRRELEVANT = "#D32F2F"
COLOR_NAME_MATCH_YES = COLOR_GREEN_RELEVANT
COLOR_NAME_MATCH_NO = COLOR_RED_IRRELEVANT


# --- Helper function to clean LLM output ---
def clean_llm_output(text: str) -> str:
    if not text: return ""
    cleaned = re.sub(r"^\s*[\*\-]\s+", "", text.strip())
    return cleaned

# --- Groq Communication ---
def query_groq_chat(messages, model_name=GROQ_DEFAULT_MODEL, temperature=0.2, max_tokens=1000):
    if not GROQ_API_KEY:
        st.error("Groq API Key is not configured in the code.")
        return None
    
    headers = { "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json" }
    payload = { "model": model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return clean_llm_output(content)
    except requests.exceptions.RequestException as e:
        st.error(f"Groq API Connection Error: {e}")
        if e.response is not None: st.error(f"Groq API Response: {e.response.text}")
        return None
    except (KeyError, IndexError, TypeError) as e:
        st.error(f"Groq API Error: Unexpected response format or issue. {e}")
        if 'response_data' in locals(): st.error(f"Full response: {response_data}")
        return None
    except Exception as e:
        st.error(f"Groq query_groq_chat unexpected error: {e}")
        return None

# --- Text Normalization for Exact Comparison ---
def normalize_for_exact_comparison(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Helper functions to extract data from a CASE dictionary ---
def get_party_names_with_source_from_case(case_dict: dict) -> list[tuple[str, str, str]]:
    names_with_source = []
    processed_normalized_names = set()

    def add_name_with_source(raw_name_str, source_desc):
        if isinstance(raw_name_str, str) and raw_name_str.strip():
            norm_name = normalize_for_exact_comparison(raw_name_str)
            if norm_name and norm_name not in processed_normalized_names:
                names_with_source.append((raw_name_str.strip(), norm_name, source_desc))
                processed_normalized_names.add(norm_name)

    if isinstance(case_dict.get("petitioner"), str): add_name_with_source(case_dict["petitioner"], "petitioner")
    if isinstance(case_dict.get("respondent"), str): add_name_with_source(case_dict["respondent"], "respondent")
    
    for party_list_key in ["gfc_petitioners", "gfc_respondents"]:
        value = case_dict.get(party_list_key)
        if isinstance(value, list):
            for i, party_obj in enumerate(value):
                if isinstance(party_obj, dict): add_name_with_source(party_obj.get("name"), f"{party_list_key}[{i}].name")
    
    gfc_orders = case_dict.get("gfc_orders_data", {})
    if isinstance(gfc_orders, dict):
        for party_list_key in ["petitioners", "respondents"]:
            for i, party_obj in enumerate(gfc_orders.get(party_list_key, [])):
                if isinstance(party_obj, dict): add_name_with_source(party_obj.get("name"), f"gfc_orders_data.{party_list_key}[{i}].name")

    raw_resp_addr = str(case_dict.get("respondent_address", ""))
    for line_num, line in enumerate(raw_resp_addr.splitlines()):
        line = line.strip()
        match = re.match(r'^\s*\d+\)\s*([^,]+?)(?=\s*(?:Ro\b|Tq\b|Dist\b|Advocate\b|$|,))', line, re.IGNORECASE)
        if match:
            potential_name = match.group(1).strip()
            if len(potential_name.split()) > 1 and len(potential_name) < 60 and any(c.isalpha() for c in potential_name):
                common_addr_indicators = ['tq', 'dist', 'road', 'street', 'nagar', 'colony', 'ro ']
                if not any(indicator in potential_name.lower() for indicator in common_addr_indicators):
                    add_name_with_source(potential_name, f"respondent_address (line approx {line_num+1}, heuristic)")
        else: # Check if line itself might be a name before "Advocate"
            name_part = line.split(' Advocate')[0].strip()
            if name_part != line and name_part and len(name_part.split()) > 1 and len(name_part) < 60:
                 add_name_with_source(name_part, f"respondent_address (line approx {line_num+1} before 'Advocate', heuristic)")
                
    return names_with_source


def get_addresses_from_case_dict_normalized(case_dict: dict) -> list:
    addresses_set = set()
    def add_address(addr_str):
        if isinstance(addr_str, str) and addr_str.strip():
            norm_addr = normalize_for_exact_comparison(addr_str)
            if norm_addr: 
                addresses_set.add(norm_addr)

    add_address(case_dict.get("petitioner_address"))
    add_address(case_dict.get("respondent_address"))
    
    for party_list_key in ["gfc_petitioners", "gfc_respondents"]:
        for party_obj in case_dict.get(party_list_key, []):
            if isinstance(party_obj, dict): add_address(party_obj.get("address"))
    
    gfc_orders = case_dict.get("gfc_orders_data", {})
    if isinstance(gfc_orders, dict):
        for party_list_key in ["petitioners", "respondents"]:
            for party_obj in gfc_orders.get(party_list_key, []):
                if isinstance(party_obj, dict): add_address(party_obj.get("address"))
    return list(addresses_set)


def compare_request_to_case(request_dict: dict, case_dict: dict) -> tuple[bool, list, list]:
    """
    Compares request fields to a single case dictionary.
    Returns: (
        primary_name_matched_in_case, 
        list_of_matched_field_details,  (each dict includes 'matched_in_case_field_source' for names)
        list_of_mismatched_field_details
    )
    """
    primary_name_matched = False
    all_party_names_with_source_in_case = get_party_names_with_source_from_case(case_dict)
    
    matched_fields_details = []
    mismatched_fields_details = []
    
    norm_req_primary_name = ""
    if "name" in request_dict: 
        norm_req_primary_name = normalize_for_exact_comparison(str(request_dict["name"]))

    if norm_req_primary_name:
        name_found_source_desc = None
        original_matched_name_in_case = norm_req_primary_name 
        for raw_name, norm_name, source_desc in all_party_names_with_source_in_case:
            if norm_name == norm_req_primary_name:
                primary_name_matched = True
                name_found_source_desc = source_desc
                original_matched_name_in_case = raw_name
                break
        
        if primary_name_matched:
            matched_fields_details.append({
                "field": "name", 
                "request_value": request_dict['name'], 
                "matched_case_value": original_matched_name_in_case,
                "matched_in_case_field_source": name_found_source_desc 
            })
        else:
            mismatched_fields_details.append({
                "field": "name", 
                "request_value": request_dict['name'], 
                "reason": f"Normalized '{norm_req_primary_name}' not found in case party names."
            })
    else:
        mismatched_fields_details.append({"field": "name", "request_value": "", "reason": "'name' field missing or empty in JSON Request."})

    for req_key, req_value_raw in request_dict.items():
        if req_key == "name": continue 

        norm_req_val = normalize_for_exact_comparison(str(req_value_raw))
        field_matched_in_this_case = False
        actual_case_value_matched = "N/A"
        match_source_description = None 
        
        if not norm_req_val and not str(req_value_raw).strip(): 
            case_val_for_empty_check_raw = case_dict.get(req_key)
            if case_val_for_empty_check_raw is None or not str(case_val_for_empty_check_raw).strip():
                field_matched_in_this_case = True
                actual_case_value_matched = ""
            else:
                 mismatched_fields_details.append({"field": req_key, "request_value": str(req_value_raw), "reason": f"Request value empty, but case has '{str(case_val_for_empty_check_raw)}'."})
        
        elif req_key == "father_name":
            found_father_name_component = False
            case_party_names_normalized = [item[1] for item in all_party_names_with_source_in_case]
            for i, norm_party_text in enumerate(case_party_names_normalized):
                if norm_req_val in norm_party_text.split(): 
                    field_matched_in_this_case = True
                    raw_name_for_context = all_party_names_with_source_in_case[i][0]
                    source_desc_for_context = all_party_names_with_source_in_case[i][2]
                    actual_case_value_matched = f"Found in party name: '{raw_name_for_context}'"
                    match_source_description = source_desc_for_context 
                    break
            if not field_matched_in_this_case:
                mismatched_fields_details.append({"field": req_key, "request_value": str(req_value_raw), "reason": "Not found as a distinct word in case party names."})
        
        elif req_key == "address":
            case_addresses_normalized = get_addresses_from_case_dict_normalized(case_dict)
            if norm_req_val in case_addresses_normalized:
                field_matched_in_this_case = True
                actual_case_value_matched = f"Matched (normalized): '{norm_req_val}'" 
            else:
                mismatched_fields_details.append({"field": req_key, "request_value": str(req_value_raw), "reason": f"Normalized address not in case addresses." })
        
        else: 
            case_val_raw = case_dict.get(req_key, "") 
            case_val_norm = normalize_for_exact_comparison(str(case_val_raw))
            if norm_req_val == case_val_norm:
                field_matched_in_this_case = True
                actual_case_value_matched = str(case_val_raw) 
            else:
                mismatched_fields_details.append({"field": req_key, "request_value": str(req_value_raw), "reason": f"Normalized '{norm_req_val}' != Case normalized '{case_val_norm}' (Original case value: '{str(case_val_raw)}')."})

        if field_matched_in_this_case:
            match_detail = {
                "field": req_key, 
                "request_value": str(req_value_raw),
                "matched_case_value": actual_case_value_matched
            }
            if match_source_description: 
                match_detail["matched_in_case_field_source"] = match_source_description
            matched_fields_details.append(match_detail)
            
    return primary_name_matched, matched_fields_details, mismatched_fields_details


# --- Core FIR Analysis Logic ---
# --- Core FIR Analysis Logic ---
def analyze_request_against_all_response_cases(request_dict, full_response_json):
    analysis_results = []
    
    if not (isinstance(request_dict, dict) and request_dict and "name" in request_dict and request_dict["name"].strip()):
        st.error("Error: JSON Request must be a non-empty object with a non-empty 'name' field.")
        return None
    if not (isinstance(full_response_json, dict) and "data" in full_response_json and 
            isinstance(full_response_json["data"], dict) and "result" in full_response_json["data"] and
            isinstance(full_response_json["data"]["result"], list)):
        st.error("Error: JSON Response structure is invalid or 'data.result' array is missing/not a list.")
        return None

    cases_in_result_array = full_response_json["data"]["result"]
    if not cases_in_result_array:
        st.info("No cases found in the 'data.result' array of the JSON Response to analyze.")
        return []

    for index, case_item_dict in enumerate(cases_in_result_array):
        if not isinstance(case_item_dict, dict): 
            st.warning(f"Item at index {index} in 'result' array is not a valid case object, skipping.")
            continue
        
        primary_name_matched, matched_fields_in_case, mismatched_fields_in_case = \
            compare_request_to_case(request_dict, case_item_dict) 
        
        is_potentially_relevant = False
        if matched_fields_in_case: 
            non_state_matches_exist = any(mf["field"] != "state" for mf in matched_fields_in_case)
            if non_state_matches_exist:
                is_potentially_relevant = True 

        potential_relevance_status_for_case = "Potentially Relevant" if is_potentially_relevant else "Potentially Not Relevant"
        
        case_identifiers_for_llm = {
            "case_name": str(case_item_dict.get("case_name", "N/A"))[:70],
            "year": case_item_dict.get("year", "N/A"),
            "court_name": str(case_item_dict.get("court_name", "N/A"))[:50],
            "petitioner": str(case_item_dict.get("petitioner", "N/A"))[:40],
            "respondent": str(case_item_dict.get("respondent", "N/A"))[:40],
            "case_link": case_item_dict.get("case_link") or case_item_dict.get("case_details_link"),
            "case_status": case_item_dict.get("case_status", "N/A"),
            "case_type_name": case_item_dict.get("case_type_name", "N/A")
        }
        
        llm_matched_fields_payload = []
        for mf in matched_fields_in_case:
            detail = {
                "request_field": mf["field"],
                "request_value": str(mf["request_value"])[:50], 
                "matched_value_in_case": str(mf["matched_case_value"])[:50] 
            }
            if mf.get('matched_in_case_field_source'):
                detail["source_in_case"] = str(mf['matched_in_case_field_source'])[:50]
            llm_matched_fields_payload.append(detail)

        llm_mismatched_fields_payload = [
            {
                "request_field": mm["field"],
                "request_value": str(mm["request_value"])[:50],
                "reason": mm["reason"][:70]
            }
            for mm in mismatched_fields_in_case
        ]

        llm_prompt_input_summary = { 
            "case_identifiers": case_identifiers_for_llm,
            "primary_name_match_status_for_case": "Matched" if primary_name_matched else "Not Matched",
            "overall_potential_relevance_for_case": potential_relevance_status_for_case,
            "details_of_matched_fields": llm_matched_fields_payload, 
            "details_of_mismatched_fields": llm_mismatched_fields_payload, 
        }

        matched_bullets_str = ""
        if llm_prompt_input_summary['details_of_matched_fields']:
            for item in llm_prompt_input_summary['details_of_matched_fields']:
                source_info = f" (found in case field: {item.get('source_in_case', 'Direct field match')})" if item.get('source_in_case') else ""
                matched_bullets_str += f"    *   The requested '{item['request_field']}' ('{item['request_value']}') matched the case's '{item['matched_value_in_case']}'{source_info}.\n"
        else:
            matched_bullets_str = "    *   No data points from the request found an exact match in this case.\n"

        mismatched_bullets_str = ""
        if llm_prompt_input_summary['details_of_mismatched_fields']:
            for item in llm_prompt_input_summary['details_of_mismatched_fields']:
                mismatched_bullets_str += f"    *   The requested '{item['request_field']}' ('{item['request_value']}') did not find an exact match (Reason: {item['reason']}).\n"
        else:
            mismatched_bullets_str = "    *   All other request fields (if any) either matched or were not applicable for mismatch reporting.\n"

        # --- REFINED REASON FOR RELEVANCE TEXT LOGIC ---
        reason_for_relevance_text_for_llm = ""
        if llm_prompt_input_summary['overall_potential_relevance_for_case'] == "Potentially Relevant":
            # This means non_state_matches_exist was True in the Python logic.
            # We count only these non-state matches from the payload for the explanation.
            actual_significant_matches_payload = [
                mf_payload for mf_payload in llm_prompt_input_summary['details_of_matched_fields'] 
                if mf_payload['request_field'] != 'state'
            ]
            count_significant_matches = len(actual_significant_matches_payload)

            if count_significant_matches > 0:
                reason_for_relevance_text_for_llm = (
                    f"This case is flagged as potentially relevant because {count_significant_matches} "
                    f"significant data point(s) (other than just 'state') from the request were found in this case record, "
                    f"as detailed in the 'Matched Data Points' section. The significance of these matches suggests a potential connection."
                )
            else:
                # This path indicates an inconsistency: Python logic determined 'Potentially Relevant' (implying non-state matches),
                # but no non-state matches were found in the llm_matched_fields_payload.
                # This could happen if `matched_fields_in_case` (from compare_request_to_case) had non-state matches,
                # but they were somehow lost or not correctly translated into `llm_matched_fields_payload`.
                # Or, it could be a very unusual edge case in data.
                reason_for_relevance_text_for_llm = (
                    "This case is flagged as potentially relevant based on initial criteria. "
                    "However, there's a discrepancy in identifying the specific significant non-state matches in the summary. "
                    "Please review the 'Matched Data Points' section carefully. "
                    "(If this issue persists, it might indicate an internal data processing inconsistency for this case)."
                )
        else: # overall_potential_relevance_for_case == "No Apparent Relevance"
            if llm_prompt_input_summary['details_of_matched_fields']: 
                # Some matches exist, but not enough for "Potentially Relevant" status by Python logic.
                # Check if it was a 'state'-only match.
                is_only_state_match = all(
                    mf_payload['request_field'] == 'state' for mf_payload in llm_prompt_input_summary['details_of_matched_fields']
                ) and any( # ensure there actually IS a state match
                    mf_payload['request_field'] == 'state' for mf_payload in llm_prompt_input_summary['details_of_matched_fields']
                )

                if is_only_state_match:
                    reason_for_relevance_text_for_llm = (
                        "This case is flagged as having Potentially Not Relevant because, although the 'state' field matched, "
                        "no other significant data points from the request were found in this case record. "
                        "A 'state'-only match is not considered a strong indicator of specific relevance by the system."
                    )
                else:
                    # Matches exist, but they are not 'state'-only, yet Python still deemed it "No Apparent Relevance".
                    # This means `non_state_matches_exist` was False, which implies all matches in `matched_fields_in_case`
                    # were 'state', or `matched_fields_in_case` was empty.
                    # The `llm_prompt_input_summary['details_of_matched_fields']` existing here and not being state-only
                    # while relevance is "No Apparent Relevance" is an inconsistency.
                    # However, the most direct explanation for "No Apparent Relevance" when some matches are listed is:
                    reason_for_relevance_text_for_llm = (
                        "This case is flagged as having Potentially Not Relevant. While some data points might have matched "
                        "(as shown in 'Matched Data Points'), they did not meet the criteria for a strong potential connection "
                        "(e.g., a significant non-'state' field match was required by the system but not found or not deemed sufficient)."
                    )
            else: # No matches at all in llm_prompt_input_summary['details_of_matched_fields']
                reason_for_relevance_text_for_llm = (
                    "This case is flagged as having Potentially Not Relevant because no data points from the "
                    "request were found to match in this case record."
                )
        # --- END OF REFINED REASON ---

        prompt_for_case_explanation = f"""
You are an AI legal analyst. Your task is to provide a neat, clean, and clear explanation in bullet points about the potential relevance of a specific court case to a person of interest, based on programmatic findings of data overlaps. Do NOT use the word "accused". Do not mention "programmatic" or "normalized" in your output.

Person of Interest Details (from JSON Request):
- Name: '{str(request_dict.get("name"))}'
- Father's Name: '{str(request_dict.get("father_name", "N/A"))}'
- Address: '{str(request_dict.get("address", "N/A"))[:100]}...' 
- Year: '{str(request_dict.get("year", "N/A"))}'
- State: '{str(request_dict.get("state", "N/A"))}'
(Other request fields: {json.dumps({k: str(v)[:30] for k, v in request_dict.items() if k not in ['name', 'father_name', 'address', 'year', 'state']}, indent=2)})

Comparison Findings for Case {index + 1} (Summary):
- Case Identifiers: {json.dumps(llm_prompt_input_summary['case_identifiers'], indent=4)}
- Primary Name Match Status: {llm_prompt_input_summary['primary_name_match_status_for_case']}
- Overall Potential Relevance: {llm_prompt_input_summary['overall_potential_relevance_for_case']}
- Matched Fields Count: {len(llm_prompt_input_summary['details_of_matched_fields'])}
- Mismatched Fields Count: {len(llm_prompt_input_summary['details_of_mismatched_fields'])}

Explanation for Case {index + 1} (Strictly use bullet points for each section below. Be factual and clear.):

*   **Primary Name Assessment:**
    *   The name '{request_dict.get('name')}' provided in the request was **{llm_prompt_input_summary['primary_name_match_status_for_case']}** with a party name in this case.
    {(f"    *   It matched with: '{llm_prompt_input_summary['details_of_matched_fields'][0]['matched_value_in_case']}' (found in case field: {llm_prompt_input_summary['details_of_matched_fields'][0].get('source_in_case', 'N/A')}).") if primary_name_matched and llm_prompt_input_summary['details_of_matched_fields'] and llm_prompt_input_summary['details_of_matched_fields'][0]['request_field'] == 'name' else ""}

*   **Matched Data Points from Request in this Case:**
{matched_bullets_str}
*   **Reason for Potential Relevance ({llm_prompt_input_summary['overall_potential_relevance_for_case']}):**
    *   {reason_for_relevance_text_for_llm}

*   **Case Description:**
    *   Case: {case_identifiers_for_llm['case_name']} ({case_identifiers_for_llm['year']})
    *   Court: {case_identifiers_for_llm['court_name']}
    *   Type: {case_identifiers_for_llm['case_type_name']}
    *   Status: {case_identifiers_for_llm['case_status']}
    *   Petitioner(s): {case_identifiers_for_llm['petitioner']}
    *   Respondent(s): {case_identifiers_for_llm['respondent']}

*   **Case Link:**
    *   Link: {case_identifiers_for_llm['case_link'] if case_identifiers_for_llm['case_link'] else 'Not available'}

*   **Mismatched Data Points from Request in this Case (for context):**
{mismatched_bullets_str}
"""
                # --- MODIFICATION: AUTOMATICALLY FETCH LLM EXPLANATION FOR RELEVANT CASES ---
        llm_explanation_content_auto_fetched = None # Initialize
        if potential_relevance_status_for_case == "Potentially Relevant":
            if GROQ_API_KEY and GROQ_API_KEY != "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": # Check against placeholder
                # This call happens during the main analysis phase for relevant cases
                explanation = query_groq_chat(
                    messages=[{"role": "user", "content": prompt_for_case_explanation}]
                )
                if explanation is not None and explanation.strip():
                    llm_explanation_content_auto_fetched = explanation
                elif explanation == "": # Explicitly empty response
                    llm_explanation_content_auto_fetched = "LLM returned an empty explanation."
                else: # query_groq_chat returned None (API error or other issue)
                    llm_explanation_content_auto_fetched = "LLM explanation fetch failed during analysis (API error or unexpected issue)."
            else:
                llm_explanation_content_auto_fetched = "Groq API Key not configured or is placeholder; LLM explanation skipped during analysis."
        # --- END OF MODIFICATION ---
        analysis_results.append({
            "record_source": f"Case ({index + 1})",
            "primary_name_matched_status": "Yes" if primary_name_matched else "No",
            "overall_relevance": potential_relevance_status_for_case, 
            "llm_explanation_prompt_messages": [{"role": "user", "content": prompt_for_case_explanation}],
            "case_details_for_display": case_item_dict, 
        })
    return analysis_results

# --- Streamlit App (UI part) ---
st.set_page_config(layout="wide", page_title="Ecourt Analyser")
st.markdown(f"<h1 style='color: {COLOR_DARK_BLUE};'>📊 Ecourt Analyser</h1>", unsafe_allow_html=True)

default_values = {
    "request_json_input_str_main": "", "response_json_input_str_main": "",
    "request_json_parsed": None, "response_json_full": None,
    "fir_analysis_results": None, "llm_explanations_for_cases": {},
}
for key, value in default_values.items():
    if key not in st.session_state: st.session_state[key] = value

TEXT_AREA_HEIGHT = 280
input_col1, input_col2 = st.columns(2)

with input_col1:
    st.markdown(f"<h3 style='color: {COLOR_DARK_BLUE};'>JSON Request</h3>", unsafe_allow_html=True)
    st.session_state.request_json_input_str_main = st.text_area(
        "Paste JSON Request", value=st.session_state.request_json_input_str_main, height=TEXT_AREA_HEIGHT,
        key="request_json_input_main_ta", help="Details of the person of interest. Must contain a 'name' field."
    )

with input_col2:
    st.markdown(f"<h3 style='color: {COLOR_DARK_BLUE};'>JSON Response</h3>", unsafe_allow_html=True)
    st.session_state.response_json_input_str_main = st.text_area(
        "Paste JSON Response", value=st.session_state.response_json_input_str_main, height=TEXT_AREA_HEIGHT,
        key="response_json_input_main_ta", help="Expected: {'data': {'result': [...]}} containing a list of cases."
    )

if st.button("Analyze Cases", type="primary", use_container_width=True, key="analyze_main_button"):
    if not GROQ_API_KEY:
        st.error("Groq API Key is missing from the code.")
    else:
        st.session_state.update({
            "request_json_parsed": None, "response_json_full": None,
            "fir_analysis_results": None, "llm_explanations_for_cases": {}
        })
        
        current_request_str = st.session_state.request_json_input_str_main
        current_response_str = st.session_state.response_json_input_str_main

        if not current_request_str.strip() or not current_response_str.strip():
            st.warning("Please provide content for both JSON Request and JSON Response.")
        else:
            parsed_request_json = None
            parsed_response_json = None
            try: parsed_request_json = json.loads(current_request_str)
            except Exception as e: st.error(f"Error Parsing 'JSON Request': {e}") 
            try: parsed_response_json = json.loads(current_response_str)
            except Exception as e: st.error(f"Error Parsing 'JSON Response': {e}")

            if parsed_request_json and parsed_response_json:
                if not (isinstance(parsed_request_json, dict) and parsed_request_json and "name" in parsed_request_json and parsed_request_json["name"].strip()):
                     st.error("The 'JSON Request' must be a non-empty JSON object with a non-empty 'name' field.")
                elif not (isinstance(parsed_response_json.get("data"), dict) and 
                          isinstance(parsed_response_json.get("data", {}).get("result"), list)):
                    st.error("Invalid JSON Response structure. Expected: {'data': {'result': [...]}} where 'result' is a list of cases.")
                else:
                    try:
                                                # --- MODIFICATION: ADD SPINNER FOR ANALYSIS + AUTO LLM FETCH ---
                        with st.spinner("Analyzing cases and fetching LLM explanations for relevant ones... This may take a moment."):
                            results = analyze_request_against_all_response_cases(parsed_request_json, parsed_response_json)
                        # --- END OF MODIFICATION ---y
                        if results is not None:
                            st.session_state.fir_analysis_results = results
                            relevant_count = sum(1 for r in results if r["overall_relevance"] == "Potentially Relevant")
                            st.success(f"Analysis complete! Processed {len(results)} case(s). Found {relevant_count} with potential relevance based on data overlaps.")
                        else: st.error("Analysis could not be completed.")
                    except Exception as e: st.error(f"Unexpected analysis error: {e}")

st.markdown("<hr>", unsafe_allow_html=True)

if st.session_state.fir_analysis_results is not None:
    st.markdown(f"<h3 style='color: {COLOR_DARK_BLUE};'>Case Analysis Table</h3>", unsafe_allow_html=True)

    column_weights = [0.8, 0.8, 1.5, 3.5] 
    header_cols = st.columns(column_weights)
    header_titles = ["Case No.", "Primary Name Matched", "Potential Relevance", "LLM Explanation & Details"]
    for col, title in zip(header_cols, header_titles): col.markdown(f"**{title}**")
    st.markdown("""<hr style="height:2px;border:none;color:#666;background-color:#666;" /> """, unsafe_allow_html=True)

    for row_data in st.session_state.fir_analysis_results:
        record_source_text = str(row_data["record_source"]) 
        sanitized_record_key_suffix = f"_{row_data['primary_name_matched_status']}_{hash(row_data['overall_relevance'])}"
        clean_record_source = re.sub(r'[^a-zA-Z0-9]', '_', record_source_text)
        sanitized_record_key = f"{clean_record_source}{sanitized_record_key_suffix}"


        row_cols = st.columns(column_weights)

        with row_cols[0]: st.markdown(record_source_text) 
        with row_cols[1]: 
            name_match_status = row_data['primary_name_matched_status']
            color_name = COLOR_NAME_MATCH_YES if name_match_status == "Yes" else COLOR_NAME_MATCH_NO
            st.markdown(f"<span style='color:{color_name};'>{name_match_status}</span>", unsafe_allow_html=True)

        with row_cols[2]: 
            relevance_text = row_data['overall_relevance']
            color_relevance = COLOR_RED_IRRELEVANT 
            if relevance_text == "Potentially Relevant": 
                color_relevance = COLOR_GREEN_RELEVANT
            st.markdown(f"<span style='color:{color_relevance}; font-weight:bold;'>{relevance_text}</span>", unsafe_allow_html=True)
        
        # --- MODIFICATION: UI LOGIC FOR DISPLAYING LLM EXPLANATIONS ---
        with row_cols[3]: 
            is_relevant_case = row_data['overall_relevance'] == "Potentially Relevant"
            explanation_to_display = None
            expander_expanded_by_default = False
            show_manual_fetch_button = False
            manual_fetch_button_text = "Explanation"

            if is_relevant_case:
                # For relevant cases, check if explanation was auto-fetched
                if row_data.get("llm_explanation_content"):
                    explanation_to_display = row_data["llm_explanation_content"]
                    # Check if it's a success message or an actual explanation
                    if not ("fetch failed" in explanation_to_display.lower() or \
                            "skipped during analysis" in explanation_to_display.lower() or \
                            "empty explanation" in explanation_to_display.lower()):
                        expander_expanded_by_default = True # Expand if successfully auto-fetched
                    else: # Auto-fetch had an issue, allow manual retry
                        show_manual_fetch_button = True
                        manual_fetch_button_text = "Retry  Explanation"
                else: # Should not happen if auto-fetch logic is robust, but as a fallback
                    show_manual_fetch_button = True
                    manual_fetch_button_text = "Fetch  Explanation (Auto-fetch missing)"
            else:
                # For non-relevant cases, always show the button for on-demand fetching
                show_manual_fetch_button = True
                manual_fetch_button_text = "Explanation"

            # Display manually fetched explanation if it exists in session_state
            # This overrides auto-fetched if user clicks button
            if record_source_text in st.session_state.llm_explanations_for_cases:
                explanation_to_display = st.session_state.llm_explanations_for_cases[record_source_text]
                expander_expanded_by_default = True # If button was clicked, expand

            # Display the expander if there's content to show
            if explanation_to_display:
                with st.expander(f"Explanation for {record_source_text}", expanded=expander_expanded_by_default): 
                    st.markdown(explanation_to_display, unsafe_allow_html=True)
            
            # Display the button if needed
            if show_manual_fetch_button and row_data["llm_explanation_prompt_messages"]:
                # Ensure button is not shown if a successful auto-fetched explanation is already displayed and expanded
                # (This condition might be redundant if `show_manual_fetch_button` logic is perfect, but adds safety)
                if not (is_relevant_case and explanation_to_display and expander_expanded_by_default and \
                        not ("fetch failed" in explanation_to_display.lower() or \
                             "skipped during analysis" in explanation_to_display.lower() or \
                             "empty explanation" in explanation_to_display.lower())):

                    explain_button_key = f"explain_btn_{sanitized_record_key}" # Key must be unique
                    if st.button(manual_fetch_button_text, key=explain_button_key, help=f"LLM explanation for {record_source_text}", type="secondary"):
                        if not GROQ_API_KEY or GROQ_API_KEY == "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": # Check placeholder
                            st.error("Groq API Key is missing or is placeholder.")
                        else:
                            with st.spinner(f"Asking Groq ({GROQ_DEFAULT_MODEL})..."):
                                explanation = query_groq_chat(row_data["llm_explanation_prompt_messages"])
                                if explanation is not None and explanation.strip():
                                    st.session_state.llm_explanations_for_cases[record_source_text] = explanation
                                elif explanation == "":
                                    st.session_state.llm_explanations_for_cases[record_source_text] = "LLM returned an empty explanation."
                                else:
                                    st.session_state.llm_explanations_for_cases[record_source_text] = "LLM explanation fetch failed or was empty."
                                st.rerun() # Rerun to update the display with the new explanation
        # --- END OF MODIFICATION ---
            
            # 3. Modification: Remove "Raw JSON" Expander
            # with st.expander(f"Raw JSON for {record_source_text}", expanded=False):
            #     st.json(row_data["case_details_for_display"], expanded=False)
        
        st.markdown("""<hr style="height:1px;border:none;color:#eee;background-color:#eee;" />""", unsafe_allow_html=True)

elif not st.session_state.request_json_input_str_main.strip() and \
     not st.session_state.response_json_input_str_main.strip():
    st.info("Enter JSON data in the fields above and click 'Analyze Cases' to begin.")

