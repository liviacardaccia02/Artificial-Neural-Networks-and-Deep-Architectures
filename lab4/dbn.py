from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 500
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        vis = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        _, hid = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis)
        _, pen = self.rbm_stack['hid--pen'].get_h_given_v_dir(hid)
        vis_top = np.concatenate((pen, lbl), axis=1)
        print(f"vis_top shape: {vis_top.shape}, pen shape: {pen.shape}, lbl shape: {lbl.shape}")
        for _ in range(self.n_gibbs_recog):
            _, h_top = self.rbm_stack["pen+lbl--top"].get_h_given_v(vis_top)
            _, vis_top_new = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_top)
            
            # Clamping: manteniamo fisse le feature dell'immagine estratte al passo 1
            # e aggiorniamo unicamente le probabilità/attivazioni delle etichette
            vis_top[:, -self.sizes["lbl"]:] = vis_top_new[:, -self.sizes["lbl"]:]

        predicted_lbl = vis_top[:, -self.sizes["lbl"]:]
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # 1. Inizializzazione dello strato visibile superiore
        pen_random = np.random.rand(n_sample, self.sizes["pen"])
        vis_top = np.concatenate((pen_random, lbl), axis=1)
            
        for _ in range(self.n_gibbs_gener):

            # 2. Gibbs Sampling nel Top RBM
            _, h_top = self.rbm_stack["pen+lbl--top"].get_h_given_v(vis_top)
            _, vis_top_new = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_top)
            
            # Clamping: ripristino delle etichette originali
            vis_top_new[:, -self.sizes["lbl"]:] = lbl
            vis_top = vis_top_new
            
            # 3. Estrazione delle feature e propagazione Top-Down (connessioni dirette)
            pen = vis_top[:, :-self.sizes["lbl"]]
            _, hid = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen)
            
            # 4. Calcolo del livello visibile inferiore
            p_vis, _ = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid)
            
            # Nota: sostituiamo l'array casuale con le probabilità p_vis della ricostruzione.
            # L'uso delle probabilità continue (invece delle attivazioni binarie) produce 
            # immagini sfumate molto più chiare per la visualizzazione video.
            vis = p_vis 
            
            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :
        
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """            
            self.rbm_stack["vis--hid"].cd1(visible_trainset=vis_trainset, n_iterations=n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")
            
            hid_trainset = self.rbm_stack["vis--hid"].get_h_given_v(vis_trainset)[1] # get hidden activities from visible data to train next RBM

            print ("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """            
            self.rbm_stack["hid--pen"].cd1(visible_trainset=hid_trainset, n_iterations=n_iterations)
            self.rbm_stack["vis--hid"].untwine_weights()            
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")  
            
            pen_trainset = self.rbm_stack["hid--pen"].get_h_given_v(hid_trainset)[1] # get penultimate activities from hidden data to train next RBM          

            print ("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            self.rbm_stack["hid--pen"].untwine_weights()
            pen_lbl_trainset = np.concatenate((pen_trainset,lbl_trainset),axis=1) # concatenate penultimate and label data to train top RBM
            self.rbm_stack["pen+lbl--top"].cd1(visible_trainset=pen_lbl_trainset, n_iterations=n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):            
                
                # Estrazione mini-batch casuale
                minibatch_idx = np.random.randint(0, self.n_samples, self.batch_size)
                v_wake = vis_trainset[minibatch_idx]
                lbl_wake = lbl_trainset[minibatch_idx]

                # --- 1. WAKE PHASE ---
                # Drive the network bottom to top fixing the visible and label data
                _, h_wake = self.rbm_stack["vis--hid"].get_h_given_v_dir(v_wake)
                _, pen_wake = self.rbm_stack["hid--pen"].get_h_given_v_dir(h_wake)
                
                vis_top_wake = np.concatenate((pen_wake, lbl_wake), axis=1)

                # --- 2. GIBBS SAMPLING IN TOP RBM ---
                v_top = vis_top_wake
                _, h_0_top = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_top)
                h_top = h_0_top
                
                for _ in range(self.n_gibbs_wakesleep):
                    _, v_top = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_top)
                    _, h_top = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_top)
                
                v_k_top = v_top
                h_k_top = h_top

                # --- 3. SLEEP PHASE ---
                # Da top RBM, propaga verso il basso per generare il "sogno"
                pen_sleep = v_k_top[:, :-self.sizes["lbl"]]
                _, hid_sleep = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen_sleep)
                _, vis_sleep = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid_sleep)

                # --- 4. PREDICTIONS ---
                # Predizioni Generative partendo dalle attivazioni della wake phase
                p_hid_gen_pred, _ = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen_wake)
                p_vis_gen_pred, _ = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_wake)

                # Predizioni di Riconoscimento partendo dalle attivazioni della sleep phase
                p_hid_rec_pred, _ = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_sleep)
                p_pen_rec_pred, _ = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid_sleep)
                
                # --- 5. UPDATE PARAMETERS ---
                # Aggiorna la Top RBM usando CD (sfruttando il campionamento ottenuto al punto 2)
                self.rbm_stack["pen+lbl--top"].update_params(vis_top_wake, h_0_top, v_k_top, h_k_top)

                # Aggiorna i parametri Generativi (top-down)
                self.rbm_stack["hid--pen"].update_generate_params(inps=pen_wake, trgs=h_wake, preds=p_hid_gen_pred)
                self.rbm_stack["vis--hid"].update_generate_params(inps=h_wake, trgs=v_wake, preds=p_vis_gen_pred)

                # Aggiorna i parametri di Riconoscimento (bottom-up)
                self.rbm_stack["vis--hid"].update_recognize_params(inps=vis_sleep, trgs=hid_sleep, preds=p_hid_rec_pred)
                self.rbm_stack["hid--pen"].update_recognize_params(inps=hid_sleep, trgs=pen_sleep, preds=p_pen_rec_pred)

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
