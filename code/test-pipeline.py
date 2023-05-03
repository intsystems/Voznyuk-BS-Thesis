from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.datatypes import Segmentation
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
from inpladesys.models.pipeline_author_diarizer import PipelineAuthorDiarizer
from inpladesys.models.clustering.auto_k_means import AutoKMeans
from inpladesys.util.cacher import Cacher
import tensorflow as tf
import numpy as np
np.seterr(all='print')
np.seterr(divide='print')
tf.compat.v1.enable_eager_execution()

def evaluate(params: dict, dataset_index: int, cache_dir=None, linear=True, test=False):
    features_file_name = 'features_files/cng-sw-bow.json'
    random_state = 0xBeeFeed
    pl_params = dict()
    pl_params['basic_feature_extractor'] = BasicFeatureExtractor(
        features_file_name=features_file_name,
        context_size=params['context_size'])

    pl_params['feature_transformer'] = GroupRepelFeatureTransformer(
    output_dimension=params['gr_output_dimension'],
            reinitialize_on_fit=False,
            nonlinear_layer_count=params['gr-nonlinear_layer_count'],
            iteration_count=params['gr-iteration_count'],
            learning_rate=1e-5,
            random_state=random_state)
    pl_params['clusterer'] = AutoKMeans(2, 2) if dataset_index != 2 else AutoKMeans(2, 2)
    if params['basic_feature_extender'] == 'f**2':
        pl_params['basic_feature_extender'] = lambda f: np.concatenate((f, f ** 2), axis=0)
    else:
        pl_params['basic_feature_extender'] = None
    keys = ['context_size', 'basic_feature_extender']
    if cache_dir is None: cache_dir = ""
    cache_dir = ".model-cache/multi-pipeline-5/task-{}/".format("abc"[dataset_index]) + cache_dir
    if cache_dir is None:
        cache_dir += ''.join(
            "({}={})".format(k, params[k]) for k in keys)
    pad = PipelineAuthorDiarizer(pl_params, cacher=Cacher(cache_dir))

    path_to_train_dataset = "../generated_documents_multiclass_preprocessed_5"
    # path_to_train_dataset = "../generated_documents_1/train"
    dataset = Pan16DatasetLoader(path_to_train_dataset).load_dataset()
    dataset.shuffle(random_state=random_state)
    train_validate_data, test_data = dataset.split(0, int(dataset.size * 0.7))
    train_data, validate_data = train_validate_data.split(0, int(train_validate_data.size * 0.7))
    number_of_document_for_check = int(train_validate_data.size)
    print(number_of_document_for_check)
    if test:
        train_data = train_validate_data
        validate_data = test_data

    print("Training...")
    pad.train(train_data)

    hs = pad.predict(validate_data.documents,
                     author_counts=[s.author_count for d, s in validate_data] if dataset_index == 1 else None)
    ys = validate_data.segmentations

params = dict()
params['context_size'] = 120
params['gr_output_dimension'] = 40
params['gr-nonlinear_layer_count'] = 0
params['gr-iteration_count'] = 40
params['basic_feature_extender'] = 'f2'

evaluate(params,
         1,
         cache_dir="cng120-bow120-sw--ctx{}--f2-s".format(
             params['context_size'],
             1 if params['basic_feature_extender'] == 'f2' else 0),
         linear=True,
         test=True
)
