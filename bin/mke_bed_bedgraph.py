#convert bedgraphs to more 'bed' like files
import numpy as np
import pandas as pd

annots = ['Nott19_Microglia_h3k27ac','Nott19_Microglia_atac','pred_Nott19_Microglia_h3k27ac',
          'Nott19_Neuron_h3k27ac','Nott19_Neuron_atac','pred_Nott19_Neuron_h3k27ac',
          'Nott19_Oligodendrocyte_h3k27ac','Nott19_Oligodendrocyte_atac','pred_Nott19_Oligodendrocyte_h3k27ac'
          ]
all_counts = dict((l,0) for l in annots)

peak_cutoff = np.arcsinh(1.0)
model_peak_cutoff = peak_cutoff
for annots_i in annots:
    print(annots_i)
    annots_count = []
    for chr_i in range(1,23):
        a = pd.read_csv(f'./annots/{annots_i}.{chr_i}.bedGraph',sep='\t',header=None)
        if annots_i in ['Nott19_Microglia_h3k27ac','Nott19_Microglia_atac',
                        'Nott19_Neuron_h3k27ac','Nott19_Neuron_atac',
                        'Nott19_Oligodendrocyte_h3k27ac','Nott19_Oligodendrocyte_atac',
                        'Nott19_Astrocyte_h3k27ac','Nott19_Astrocyte_atac',]:
            a = a[a[3]>peak_cutoff]
        else:
            a = a[a[3]>model_peak_cutoff]
        a = a[[0,1,2]].reset_index(drop=True)
        a.to_csv(f'./annots/{annots_i}.{chr_i}.bed',sep='\t',header=None,index=False)
        print("chr",chr_i,"nrow:",a.shape[0])
        #save num bed peaks
        annots_count.append(a.shape[0])
    all_counts[annots_i]=np.sum(annots_count)

df_counts = pd.DataFrame(all_counts.items(), columns=['bed', 'count'])
df_counts.to_csv("./annots/bed_peak_counts.csv", sep=',',index=False)
