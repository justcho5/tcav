import sys
sys.path.append("/Users/justina/Documents/EPFL/thesis/projects/hnsc/histoXai/tcav")
sys.path.append("/home/hjcho/projects/hnsc/histoXai/tcav")

import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.utils_plot as utils_plot # utils_plot requires matplotlib
import os
import tensorflow as tf


class TCAVProj():
    def __init__(self, project_dir, model_to_run, bottlenecks, alphas, target, concepts, graph_path, labels_path):
        self.project_dir = project_dir
        self.model_to_run = model_to_run
        self.bottlenecks = bottlenecks
        self.alphas = alphas
        self.target = target
        self.concepts = concepts
        self.graph_path = graph_path
        self.labels_path = labels_path
    def make_directories(self):
        self.working_dir = os.path.join(self.project_dir , "tmp/{}".format(self.model_to_run))
        self.activation_dir =  os.path.join(self.working_dir, 'activations/')
        self.cav_dir = os.path.join(self.working_dir,'cavs/')
        self.source_dir = os.path.join(self.project_dir, "source_dir")
        print(self.working_dir)
        print(self.activation_dir)
        print(self.cav_dir)
        print(self.source_dir)
        utils.make_dir_if_not_exists(self.activation_dir)
        utils.make_dir_if_not_exists(self.working_dir)
        utils.make_dir_if_not_exists(self.cav_dir)
    def run(self):
        self.sess = utils.create_session()
        if self.model_to_run == 'InceptionV3':
            self.mymodel=model.InceptionV3Wrapper_public(self.sess, self.graph_path, self.labels_path)
        if self.model_to_run == 'GoogleNet':
            self.mymodel = model.GoolgeNetWrapper_public(self.sess, self.graph_path, self.labels_path)
        if self.model_to_run == 'XceptionHPV':
            self.mymodel = model.XceptionHPVWrapper_public(self.sess, self.graph_path, self.labels_path)
        act_generator = act_gen.ImageActivationGenerator(self.mymodel, self.source_dir, self.activation_dir, max_examples=100)
        tf.logging.set_verbosity(0)
        mytcav = tcav.TCAV(self.sess,
                           self.target,
                           self.concepts,
                           self.bottlenecks,
                           act_generator,
                           self.alphas,
                           cav_dir=self.cav_dir,
                           num_random_exp=10)
        print ('This may take a while... Go get coffee!')
        results = mytcav.run(run_parallel=True)
        print ('done!')

        # returns dictionary of plot data
        plot_data = utils_plot.plot_results(results, os.path.join(self.project_dir, 'results/inceptionv3_tcav.png'), num_random_exp=10)
def main():
    # project_dir = '/Users/justina/Documents/EPFL/thesis/projects/hnsc/histoXai/ace_small_test'
    # project_dir = '/Users/justina/Documents/EPFL/thesis/projects/hnsc/histoXai/imagenet_small_test'
    project_dir = '/home/hjcho/projects/hnsc/histoXai/imagenet_small_test'
    model_to_run = 'InceptionV3'
    # bottlenecks = [ 'mixed4c']
    bottlenecks= ['mixed0_2', 'mixed1_2', 'mixed2_2','mixed3_2','mixed4_2','mixed5_2', 'mixed6_2','mixed7_2','mixed8_2','mixed9_3','mixed10_2']
    # bottlenecks=['add_11/add','add_10/add','add_9/add','add_8/add','add_7/add','add_6/add']
    # bottlenecks=['add_9/add']
    # bottlenecks=['mixed10_2']
    alphas = [0.1]
    target = 'zebra'
    concepts = ["dotted","striped","zigzagged"]
    GRAPH_PATH=os.path.join(project_dir,"graphs/inceptionv3.pb")
    # LABEL_PATH = os.path.join(project_dir, "graphs/labels.txt")
    LABEL_PATH = os.path.join(project_dir, "graphs/imagenet_comp_graph_label_strings.txt")

    run = TCAVProj(project_dir, model_to_run, bottlenecks, alphas, target, concepts, GRAPH_PATH, LABEL_PATH)
    run.make_directories()
    run.run()

if __name__ == '__main__':
    main()
