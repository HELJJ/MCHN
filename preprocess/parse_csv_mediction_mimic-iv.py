import os
from datetime import datetime
from collections import OrderedDict

import pandas
import pandas as pd
import numpy as np


class EHRParser:
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_time_col = 'adm_time'
    cid_col = 'cid'
    NDC = 'ndc'
    DRUG = 'drug'
    ATC3 = 'atc3'
    TEXT = 'TEXT'
    CATEGORY = 'CATEGORY'




    def __init__(self, path):
        self.path = path

        self.skip_pid_check = False

        self.patient_admission = None
        self.admission_codes = None
        self.admission_procedures = None
        self.admission_medications = None
        self.medicine_code = None
        self.patient_note = None
        self.rxnorm2RXCUI_file = '/home/zhangxin21/chet_second/data/mimic3/raw/ndc2rxnorm_mapping.txt' #ndc2rxnorm_mapping.txt ndc2atc_level4.csv
        self.drugbankinfo = '/home/zhangxin21/chet_second/data/mimic3/raw/drugbank_drugs_info.csv'
        self.RXCUI2atc4_file = '/home/zhangxin21/chet_second/data/mimic3/raw/ndc2atc_level4.csv'
        self.parse_fn = {'d': self.set_diagnosis}

    def set_admission(self):
        raise NotImplementedError

    def set_diagnosis(self):
        raise NotImplementedError

    def set_mediction(self):
        raise NotImplementedError

    @staticmethod
    def to_standard_icd9(code: str):
        raise NotImplementedError

    def parse_admission(self):
        print('parsing the csv file of admission ...')
        filename, cols, converters = self.set_admission()
        admissions = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        admissions = self._after_read_admission(admissions, cols)
        all_patients = OrderedDict()
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
            pid, adm_id, adm_time = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_time_col]]
            if pid not in all_patients:
                all_patients[pid] = []
            admission = all_patients[pid]
            admission.append({self.adm_id_col: adm_id, self.adm_time_col: adm_time})
        print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

        patient_admission = OrderedDict()
        for pid, admissions in all_patients.items():
            if len(admissions) >= 2:
                patient_admission[pid] = sorted(admissions, key=lambda admission: admission[self.adm_time_col])

        self.patient_admission = patient_admission

    # ATC3-to-drugname
    def ATC3toDrug(self, med_pd):
        atc3toDrugDict = {}
        for atc3, drugname in med_pd[['ATC3', 'drug']].values:
            if atc3 in atc3toDrugDict:
                atc3toDrugDict[atc3].add(drugname)
            else:
                atc3toDrugDict[atc3] = set(drugname)

        return atc3toDrugDict

    def atc3toSMILES(self, ATC3toDrugDict, druginfo):
        drug2smiles = {}
        atc3tosmiles = {}
        for drugname, smiles in druginfo[['name', 'moldb_smiles']].values:
            if type(smiles) == type('a'):
                drug2smiles[drugname] = smiles
        for atc3, drug in ATC3toDrugDict.items():
            temp = []
            for d in drug:
                try:
                    temp.append(drug2smiles[d])
                except:
                    pass
            if len(temp) > 0:
                atc3tosmiles[atc3] = temp[:3]

        return atc3tosmiles


    # medication mapping
    def codeMapping2atc4(self, med_pd):
        rxnorm2RXCUI_file = self.rxnorm2RXCUI_file
        #RXCUI2atc4_file = '../data/mimic3/raw/ndc2atc_level4.csv' #
        with open(rxnorm2RXCUI_file, 'r') as f:
            rxnorm2RXCUI = eval(f.read())
        # rxnorm2RXCUI_df = pd.read_csv(rxnorm2RXCUI_file)
        # rxnorm2RXCUI = pd.Series(rxnorm2RXCUI_df.RXCUI.values, index=rxnorm2RXCUI_df.NDC).to_dict()
        med_pd['RXCUI'] = med_pd['ndc'].map(rxnorm2RXCUI)
        med_pd.dropna(inplace=True)

        rxnorm2atc4 = pd.read_csv(self.RXCUI2atc4_file)
        rxnorm2atc4 = rxnorm2atc4.drop(columns=['YEAR', 'MONTH', 'NDC'])
        rxnorm2atc4.drop_duplicates(subset=['RXCUI'], inplace=True)
        med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

        med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
        med_pd = med_pd.reset_index(drop=True)
        med_pd = med_pd.merge(rxnorm2atc4, on=['RXCUI'])
        med_pd.drop(columns=['ndc', 'RXCUI'], inplace=True)
        med_pd['ATC4'] = med_pd['ATC4'].map(lambda x: x[:4])
        med_pd = med_pd.rename(columns={'ATC4': 'ATC3'})
        med_pd = med_pd.drop_duplicates()
        med_pd = med_pd.reset_index(drop=True)
        return med_pd


    def parse_medicition(self):
        print('parsing the csv file of mediction ...')
        filename, cols, converters = self.set_mediction()
        med_pd = pd.read_csv(os.path.join(self.path, filename), dtype={'ndc': 'category'})
        #med_pd = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        med_pd.drop(columns=['drug_type', 'pharmacy_id', 'prod_strength',
                        'form_rx','dose_val_rx','dose_unit_rx',
                        'form_val_disp','form_unit_disp','doses_per_24_hrs', 'gsn',
                        'route','stoptime', 'starttime'], axis=1, inplace=True)
        med_pd.drop(index=med_pd[med_pd['ndc'] == '0'].index, axis=0, inplace=True)
        med_pd.fillna(method='pad', inplace=True)
        med_pd.dropna(inplace=True)
        med_pd.drop_duplicates(inplace=True)
        # med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
        # med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
        med_pd.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
        med_pd = med_pd.reset_index(drop=True)
        med_pd = self.codeMapping2atc4(med_pd)
        # med to SMILES mapping
        atc3toDrug = self.ATC3toDrug(med_pd)
        drugbankinfo = self.drugbankinfo
        druginfo = pd.read_csv(drugbankinfo)
        atc3toSMILES = self.atc3toSMILES(atc3toDrug, druginfo)
        #dill.dump(atc3toSMILES, open(atc3toSMILES_file, 'wb'))
        med_pd = med_pd[med_pd.ATC3.isin(atc3toSMILES.keys())]
        print('complete medication processing')
        #将小于两次的就诊删掉
        # 计算每个 subject_id 的 hadm_id 数量
        hadm_counts = med_pd.groupby('subject_id')['hadm_id'].nunique()
        # 找到 hadm_id 数量大于2的 subject_id
        valid_subject_ids = hadm_counts[hadm_counts > 2].index
        # 过滤掉 hadm_id 数量小于或等于2的数据
        med_pd = med_pd[med_pd['subject_id'].isin(valid_subject_ids)]
        #因为按照atc3来作为药物的code进行索引的，所以只需要atc3
        result = OrderedDict()
        for i, row in med_pd.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(med_pd)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code = row['hadm_id'], row['ATC3'] #只选择对应的hadm_id和atc3
                if code == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                codes = result[adm_id]
                codes.append(code)
        print('\r\t%d in %d rows' % (len(med_pd), len(med_pd)))
        del_adm_id = []
        for patient_adm_id, diagnosis_codes in result.items():
            if len(diagnosis_codes) < 2:
                del_adm_id.append(patient_adm_id)
        for del_id in del_adm_id:
            del result[del_id]

        self.medicine_code = result

    def parse_notes(self,use_summary=False):
        #先把clinical text整理，然后与admission中的adm_id取交集，
        # 将admission中不在clinical text中的adm_id取出来，
        # 删掉admission中的adm_id对应的数据、以及drug中的adm_id对应的数据以及patient中的adm_id对应的，当然这边要判断patient中删掉adm_id剩下是不是就诊记录>=2
        print('parsing NOTEEVENTS.csv ...')
        filename,cols,converter = self.set_note()
        notes = pd.read_csv(
            os.path.join(self.path, filename), usecols=list(cols.values()), converters=converter)


        # notes_path = os.path.join(self.path, 'NOTEEVENTS.csv')
        # notes = pd.read_csv(
        #     notes_path,
        #     usecols=['HADM_ID', 'TEXT', 'CATEGORY'], #np.str
        #     converters={'HADM_ID': lambda x: int(x) if x != '' else -1, 'TEXT': str, 'CATEGORY': str}
        # )
        result = OrderedDict()
        adm_id_list = list(self.admission_codes.keys())
        for i, row in notes.iterrows():
            adm_id = row[cols[self.adm_id_col]]
            note = [row['TEXT'] for _, row in notes[notes['HADM_ID'].isin(adm_id_list)].iterrows()
                    if row['CATEGORY'] != 'Discharge summary']
            note = ' '.join(note)
            if adm_id in result:
                print("1111s")
                print('adm_id is in the result')
            if len(note) > 0:
                result[adm_id] = note
        '''
        med_code_list = self.medicine_code
        for adm_id, text in result:
            if adm_id not in med_code_list:
                print("%s is not in med_code_list",adm_id)
        '''
        # for i, (pid, admissions) in enumerate(self.patient_admission.items()):
        #     print('\r\t%d in %d patients' % (i + 1, len(self.patient_admission)), end='')
        #     #admission_id = admissions[-1]['admission_id']
        #
        #     if use_summary:
        #         note = [row['TEXT'] for _, row in notes[notes['HADM_ID'] == admission_id].iterrows()
        #                 if row['CATEGORY'] == 'Discharge summary']
        #     else:
        #         # note = notes[notes['HADM_ID'] == admission_id]['TEXT'].tolist()
        #         note = [row['TEXT'] for _, row in notes[notes['HADM_ID'] == admission_id].iterrows()
        #                 if row['CATEGORY'] != 'Discharge summary']
        #     note = ' '.join(note)
        #     if len(note) > 0:
        #         patient_note[pid] = note
        #print('\r\t%d in %d patients' % (len(self.patient_admission), len(self.patient_admission)))
        self.patient_note = result

    def _after_read_admission(self, admissions, cols):
        return admissions

    def _parse_concept(self, concept_type):
        assert concept_type in self.parse_fn.keys()
        filename, cols, converters = self.parse_fn[concept_type]()
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        concepts = self._after_read_concepts(concepts, concept_type, cols)
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code = row[cols[self.adm_id_col]], row[cols[self.cid_col]]
                if code == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                codes = result[adm_id]
                codes.append(code)
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        del_adm_id = []
        for patient_adm_id, diagnosis_codes in result.items():
            if len(diagnosis_codes) < 2:
                del_adm_id.append(patient_adm_id)
        for del_id in del_adm_id:
            del result[del_id]

        return result

    def _after_read_concepts(self, concepts, concept_type, cols):
        return concepts

    def parse_diagnoses(self):
        print('parsing csv file of diagnosis ...')
        self.admission_codes = self._parse_concept('d')

    def calibrate_patient_by_admission(self):
        print('calibrating patients by admission ...')
        del_pids = []
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes or adm_id not in self.medicine_code:
                    break
            else:
                continue
            del_pids.append(pid)
        '''
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.medicine_code:
                    break
            else:
                continue
            del_pids.append(pid)
        '''
        for pid in set(del_pids):
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.medicine_code]:
                    if adm_id in concepts:
                        del concepts[adm_id]
            del self.patient_admission[pid]






    def calibrate_admission_by_patient(self):
        print('calibrating admission by patients ...')
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])
        del_adm_ids = [adm_id for adm_id in self.admission_codes if adm_id not in adm_id_set]
        del_adm_ids_medicine = [adm_id for adm_id in self.medicine_code if adm_id not in adm_id_set]
        del_adm_ids = del_adm_ids+del_adm_ids_medicine
        if del_adm_ids != []:
            for adm_id in set(del_adm_ids):
                if adm_id in self.admission_codes:
                    del self.admission_codes[adm_id]
                if adm_id in self.medicine_code:
                    del self.medicine_code[adm_id]

    def sample_patients(self, sample_num, seed):
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        selected_pids = np.random.choice(keys, sample_num, False)
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}
        admission_codes = dict()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes[adm_id]
        self.admission_codes = admission_codes

    def parse(self, sample_num=None, seed=6669):
        self.parse_admission()
        self.parse_diagnoses()
        self.parse_medicition()
        #self.parse_notes()
        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()
        if sample_num is not None:
            self.sample_patients(sample_num, seed)
        return self.patient_admission, self.admission_codes, self.medicine_code

class Mimic3Parser(EHRParser):
    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code


class Mimic4Parser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.icd_ver_col = 'icd_version'
        self.icd_map = self._load_icd_map()
        self.patient_year_map = self._load_patient()

    def _load_icd_map(self):
        print('loading ICD-10 to ICD-9 map ...')
        filename = 'icd10-icd9.csv'
        cols = ['ICD10', 'ICD9']
        converters = {'ICD10': str, 'ICD9': str}
        icd_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        icd_map = {row['ICD10']: row['ICD9'] for _, row in icd_csv.iterrows()}
        return icd_map

    def _load_patient(self):
        print('loading patients anchor year ...')
        filename = 'patients.csv'
        cols = ['subject_id', 'anchor_year', 'anchor_year_group']
        converters = {'subject_id': int, 'anchor_year': int, 'anchor_year_group': lambda cell: int(str(cell)[:4])}
        patient_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        patient_year_map = {row['subject_id']: row['anchor_year'] - row['anchor_year_group']
                            for i, row in patient_csv.iterrows()}
        return patient_year_map

    def set_admission(self):
        filename = 'admissions.csv'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.adm_time_col: 'admittime'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_mediction(self):
        filename = 'prescriptions.csv'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.NDC: 'ndc'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'ndc': str
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnoses_icd.csv'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {'subject_id': int, 'hadm_id': int, 'icd_code': str, 'icd_version': int}
        return filename, cols, converter

    def _after_read_admission(self, admissions, cols):
        print('\tselecting valid admission ...')
        valid_admissions = []
        n = len(admissions)
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t\t%d in %d rows' % (i + 1, n), end='')
            pid = row[cols[self.pid_col]]
            year = row[cols[self.adm_time_col]].year - self.patient_year_map[pid]
            if year > 2012:
                valid_admissions.append(i)
        print('\r\t\t%d in %d rows' % (n, n))
        print('\t\tremaining %d rows' % len(valid_admissions))
        return admissions.iloc[valid_admissions]

    def _after_read_concepts(self, concepts, concept_type, cols):
        print('\tmapping ICD-10 to ICD-9 ...')
        n = len(concepts)
        if concept_type == 'd':
            def _10to9(i, row):
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                if row[icd_ver_col] == 10:
                    if cid not in self.icd_map:
                        code = self.icd_map[cid + '1'] if cid + '1' in self.icd_map else ''
                    else:
                        code = self.icd_map[cid]
                    if code == 'NoDx':
                        code = ''
                else:
                    code = cid
                return Mimic4Parser.to_standard_icd9(code)

            cid_col, icd_ver_col = cols[self.cid_col], self.icd_ver_col
            col = np.array([_10to9(i, row) for i, row in concepts.iterrows()])
            print('\r\t\t%d in %d rows' % (n, n))
            concepts[cid_col] = col
        return concepts

    @staticmethod
    def to_standard_icd9(code: str):
        return Mimic3Parser.to_standard_icd9(code)


class EICUParser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.skip_pid_check = True

    def set_admission(self):
        filename = 'patient.csv'
        cols = {
            self.pid_col: 'patienthealthsystemstayid',
            self.adm_id_col: 'patientunitstayid',
            self.adm_time_col: 'hospitaladmitoffset'
        }
        converter = {
            'patienthealthsystemstayid': int,
            'patientunitstayid': int,
            'hospitaladmitoffset': lambda cell: -int(cell)
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnosis.csv'
        cols = {self.pid_col: 'diagnosisid', self.adm_id_col: 'patientunitstayid', self.cid_col: 'icd9code'}
        converter = {'diagnosisid': int, 'patientunitstayid': int, 'icd9code': EICUParser.to_standard_icd9}
        return filename, cols, converter

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        code = code.split(',')[0]
        c = code[0].lower()
        dot = code.find('.')
        if dot == -1:
            dot = None
        if not c.isalpha():
            prefix = code[:dot]
            if len(prefix) < 3:
                code = ('%03d' % int(prefix)) + code[dot:]
            return code
        if c == 'e':
            prefix = code[1:dot]
            if len(prefix) != 3:
                return ''
        if c != 'e' or code[0] != 'v':
            return ''
        return code

    def parse_diagnoses(self):
        super().parse_diagnoses()
        t = OrderedDict.fromkeys(self.admission_codes.keys())
        for adm_id, codes in self.admission_codes.items():
            t[adm_id] = list(set(codes))
        self.admission_codes = t
