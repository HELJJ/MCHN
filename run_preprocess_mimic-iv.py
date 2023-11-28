import os
import _pickle as pickle

from preprocess.__init__med import save_sparse, save_data
from preprocess.parse_csv_mediction_iv import  Mimic4Parser, EICUParser
from preprocess.encode_medicine import encode_code
from preprocess.build_dataset_medicine import split_patients, build_code_xy, build_heart_failure_y
from preprocess.auxiliary_med import generate_code_code_adjacent, generate_neighbors, normalize_adj, divide_middle, generate_code_levels

if __name__ == '__main__':
    conf = {

        'mimic4': {
            'parser': Mimic4Parser,
            'train_num': 8000,
            'test_num': 1000,
            'valid_num': 1000,
            'threshold': 0.01,
            'sample_num': 10000
        },
        'eicu': {
            'parser': EICUParser,
            'train_num': 8000,
            'test_num': 1000,
            'threshold': 0.01
        }
    }
    from_saved = False
    data_path = 'data'
    dataset = 'mimic4'  # mimic3, eicu, or mimic4
    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()
    parsed_path = os.path.join(dataset_path, 'parsed')
    if from_saved:
        patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
        admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
    else:
        parser = conf[dataset]['parser'](raw_path)
        sample_num = conf[dataset].get('sample_num', None)
        patient_admission, admission_codes, medicine_code = parser.parse(sample_num)
        print('saving parsed data ...')
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)
        pickle.dump(patient_admission, open(os.path.join(parsed_path, 'patient_admission.pkl'), 'wb'))
        pickle.dump(admission_codes, open(os.path.join(parsed_path, 'admission_codes.pkl'), 'wb'))
        pickle.dump(medicine_code, open(os.path.join(parsed_path, 'medicine_code.pkl'), 'wb'))

    patient_num = len(patient_admission)
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
    max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
    avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)
    print('patient num: %d' % patient_num) #6102
    print('max admission num: %d' % max_admission_num) #42
    print('mean admission num: %.2f' % avg_admission_num) #2.63
    print('max code num in an admission: %d' % max_visit_code_num) #39
    print('mean code num in an admission: %.2f' % avg_visit_code_num) #13.91

    print('encoding code ...')
    admission_codes_encoded, code_map, medicine_codes_encoded, medicine_code_map = encode_code(patient_admission, admission_codes, medicine_code)
    code_num = len(code_map)
    medicine_code_num = len(medicine_code_map)
    print('There are admission_code %d codes' % code_num) #4627
    print('There are medicine_code %d codes' % medicine_code_num) #3706

    code_levels = generate_code_levels(data_path, code_map)
    pickle.dump({
        'code_levels': code_levels,
    }, open(os.path.join(parsed_path, 'code_levels.pkl'), 'wb'))

    from sklearn.model_selection import KFold

    # Assuming your data is in a variable named `data`
    kf = KFold(n_splits=10)  # Define the number of folds
    '''
    for train_index, valid_index in kf.split(patient_admission):
        #train_data = patient_admission[train_index]
        #valid_data = patient_admission[valid_index]
        pass
        # Save these datasets to disk so they can be loaded during training
        # You can use a naming convention to differentiate between different folds
    '''
    train_pids, valid_pids = split_patients( #test_pids
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map,
        train_num=conf[dataset]['train_num'],
        #test_num=conf[dataset]['test_num']
        valid_num=conf[dataset]['valid_num']
    )
    #print('There are %d train, %d valid, %d test samples' % (len(train_pids), len(valid_pids), len(test_pids)))
    code_adj, med_code_adj = generate_code_code_adjacent(pids=train_pids, patient_admission=patient_admission,
                                           admission_codes_encoded=admission_codes_encoded, medicine_codes_encoded = medicine_codes_encoded,
                                           code_num=code_num, threshold=conf[dataset]['threshold'], med_code_num = medicine_code_num)

    common_args = [patient_admission, admission_codes_encoded, max_admission_num, code_num, medicine_codes_encoded, medicine_code_num]
    print('building train codes features and labels ...')
    (train_code_x, train_codes_y, train_visit_lens, train_med) = build_code_xy(train_pids, *common_args)
    print('building valid codes features and labels ...')
    (valid_code_x, valid_codes_y, valid_visit_lens, valid_med) = build_code_xy(valid_pids, *common_args)
    print('building test codes features and labels ...')
    #(test_code_x, test_codes_y, test_visit_lens, test_med) = build_code_xy(test_pids, *common_args)

    print('generating train neighbors ...')
    train_neighbors = generate_neighbors(train_code_x, train_visit_lens, code_adj)
    print('generating valid neighbors ...')
    valid_neighbors = generate_neighbors(valid_code_x, valid_visit_lens, code_adj)
    print('generating test neighbors ...')
    #test_neighbors = generate_neighbors(test_code_x, test_visit_lens, code_adj)

    print('generating train middles ...')
    train_divided = divide_middle(train_code_x, train_neighbors, train_visit_lens)
    print('generating valid middles ...')
    valid_divided = divide_middle(valid_code_x, valid_neighbors, valid_visit_lens)
    print('generating test middles ...')
    #test_divided = divide_middle(test_code_x, test_neighbors, test_visit_lens)

    print('building train heart failure labels ...')
    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    print('building valid heart failure labels ...')
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    print('building test heart failure labels ...')
    #test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)

    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        #'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(dataset_path, 'standard')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    #test_path = os.path.join(standard_path, 'test')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(valid_path)
        #os.makedirs(test_path)

    print('\tsaving training data')
    save_data(train_path, train_code_x, train_visit_lens, train_codes_y, train_hf_y, train_divided, train_neighbors, train_med)
    print('\tsaving valid data')
    save_data(valid_path, valid_code_x, valid_visit_lens, valid_codes_y, valid_hf_y, valid_divided, valid_neighbors, valid_med)
    print('\tsaving test data')
    #save_data(test_path, test_code_x, test_visit_lens, test_codes_y, test_hf_y, test_divided, test_neighbors, test_med)

    code_adj = normalize_adj(code_adj)
    med_code_adj = normalize_adj(med_code_adj)
    save_sparse(os.path.join(standard_path, 'code_adj'), code_adj)
    save_sparse(os.path.join(standard_path, 'med_code_adj'), med_code_adj)
