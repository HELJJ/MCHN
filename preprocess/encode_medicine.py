from collections import OrderedDict

from preprocess.parse_csv import EHRParser


def encode_code(patient_admission, admission_codes, medicine_code):
    code_map = OrderedDict()
    medicine_code_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            codes = admission_codes[admission[EHRParser.adm_id_col]]
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map)

    admission_codes_encoded = {
        admission_id: list(set(code_map[code] for code in codes))
        for admission_id, codes in admission_codes.items()
    }

    for pid, admissions in patient_admission.items():
        for admission in admissions:
            medicine_codes = medicine_code[admission[EHRParser.adm_id_col]]
            for code in medicine_codes:
                if code not in medicine_code_map:
                    medicine_code_map[code] = len(medicine_code_map)

    medicine_codes_encoded = {
        admission_id: list(set(medicine_code_map[code] for code in medicine_codes))
        for admission_id, medicine_codes in medicine_code.items()
    }

    return admission_codes_encoded, code_map, medicine_codes_encoded, medicine_code_map
