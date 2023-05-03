from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.datatypes import Segmentation
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
from inpladesys.models.pipeline_author_diarizer import PipelineAuthorDiarizer
from inpladesys.models.clustering.auto_k_means import AutoKMeans
from inpladesys.util.cacher import Cacher
import tensorflow as tf
from inpladesys.evaluation import BCubedScorer, MicroScorer, MacroScorer
import numpy as np
np.seterr(all='print')
np.seterr(divide='print')
tf.compat.v1.enable_eager_execution()

dataset_dirs = [
    "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
]


def evaluate(params: dict, dataset_index: int, cache_dir=None, linear=True, test=False):
    features_file_name = '../inpladesys/models/basic_feature_extraction/features_files/cng-sw-bow.json'
    random_state = 0xBeeFeed
    pl_params = dict()
    pl_params['basic_feature_extractor'] = BasicFeatureExtractor(
        features_file_name=features_file_name,
        context_size=params['context_size'])

    from inpladesys.models.feature_transformation import SimpleGroupRepelFeatureTransformer
    if not linear:
        pl_params['feature_transformer'] = SimpleGroupRepelFeatureTransformer(
            reinitialize_on_fit=False,
            iteration_count=params['gr-iteration_count'],
            learning_rate=5e-4,
            random_state=random_state)
    else:
        pl_params['feature_transformer'] = GroupRepelFeatureTransformer(
            output_dimension=params['gr_output_dimension'],
            reinitialize_on_fit=False,
            nonlinear_layer_count=params['gr-nonlinear_layer_count'],
            iteration_count=params['gr-iteration_count'],
            learning_rate=5e-4,
            random_state=random_state)
    print(str(pl_params['feature_transformer']))
    pl_params['clusterer'] = AutoKMeans(2, 2) if dataset_index != 2 else AutoKMeans(2, 2)
    # pl_params['clusterer'] = Deoutliizer(0.3)
    if params['basic_feature_extender'] == 'f**2':
        pl_params['basic_feature_extender'] = lambda f: np.concatenate((f, f ** 2), axis=0)
    else:
        pl_params['basic_feature_extender'] = None

    # keys = sorted(params.keys())
    keys = ['context_size', 'basic_feature_extender']
    if cache_dir is None: cache_dir = ""
    cache_dir = ".model-cache/multi-pipeline-5/task-{}/".format("abc"[dataset_index]) + cache_dir
    if cache_dir is None:
        cache_dir += ''.join(
            "({}={})".format(k, params[k]) for k in keys)
    pad = PipelineAuthorDiarizer(pl_params, cacher=Cacher(cache_dir))

    print("Loading dataset...")
    path_to_train_dataset = "../generated_documents_multiclass_preprocessed_5"
    # path_to_train_dataset = "../generated_documents_1/train"
    # path_to_train_dataset = dataset_dirs[1]
    dataset = Pan16DatasetLoader(path_to_train_dataset).load_dataset()
    # dataset.shuffle(random_state=random_state)
    train_validate_data, test_data = dataset.split(0, int(dataset.size * 0.7))
    train_data, validate_data = train_validate_data.split(0, int(train_validate_data.size * 0.7))
    number_of_document_for_check = int(train_validate_data.size)
    print(number_of_document_for_check)
    #validate_data, _ = rest_data.split(0, int(rest_data.size))

    if test:
        train_data = train_validate_data
        validate_data = test_data

    print("Training...")
    pad.train(train_data)

    # validate_data = train_data
    print("Evaluating on {} data...".format("TEST" if test else "VALIDATION"))
    hs = pad.predict(validate_data.documents,
                     author_counts=[s.author_count for d, s in validate_data] if dataset_index == 1 else None)
    ys = validate_data.segmentations

    class KScorer():
        def __init__(self, y: Segmentation, h: Segmentation):
            self.y, self.h = y, h

        def precision(self):
            return int(self.y.author_count != self.h.author_count)

        def recall(self):
            return abs(self.y.author_count - self.h.author_count)

        def f1_score(self):
            return (self.y.author_count - self.h.author_count) ** 2

    def get_scores(scorer_factory, y, h):
        scorer = scorer_factory(y, h)
        return np.array([scorer.precision(), scorer.recall(), scorer.f1_score()])

    scorer_factories = [MicroScorer, MacroScorer, BCubedScorer] if dataset_index == 0 else [BCubedScorer, KScorer]
    for sf in scorer_factories:
        score_list = [get_scores(sf, y, h) for y, h in zip(ys, hs)]
        # f1_scores = [s[2] for s in score_list]
        # print(perform_confidence_interval_test(f1_scores))
        scores = np.stack(score_list, axis=0)
        avg_scores = np.average(scores, axis=0)
        score_variances = np.var(scores, axis=0)
        print(sf)
        print(avg_scores, '+-', score_variances ** 0.5)


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