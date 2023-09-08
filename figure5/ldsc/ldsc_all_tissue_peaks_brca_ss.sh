tissues=(BREAST BRAIN BLADDER KIDNEY BREAST COLON)
tts=(BRCA GBM BLCA KIRC BRCA COAD)

#sumstats datasets -- TODO add citations and links
sumstats=(finngen_r8_C3_BREAST_EXALLC.sumstats.gz brca_gwas_onco_ss_hm3_bcac.sumstats.gz BRCA_PCP_hm3.sumstats.gz) #/illumina/scratch/deep_learning/julirsch/sumstats/pricelab_all/UKB_460K.cancer_BREAST.sumstats.gz)
gwas=(finngen bcac PCP) #UKBB_price)


for i in 0 1  #outer loop tissue
    do
        tissue=${tissues[$i]}
        tt=${tts[$i]}
        for suffix in normal #peakset
        # for suffix in tumor tumorNN tumor_non_normal NNnormal_tumor_overlap NNnormal_nontumor
        # for suffix in tumor_non_normal NNnormal_tumor_overlap NNnormal_nontumor
            do
                case $suffix in
                    tumor)
                        peakset=${tt}_peaks_${suffix}
                    ;;                    
                    normal)
                        peakset=${tissue}_peaks_${suffix}
                    ;;                    
                    tumor_non_normal)
                        peakset=${tt}_peaks_${suffix}
                    ;;
                    NNnormal_tumor_overlap)
                        peakset=${tt}_${tissue}_peaks_${suffix}
                    ;;
                    NNnormal_nontumor)
                        peakset=${tissue}_peaks_${suffix}
                    ;;
                    tumorNN)
                        peakset=${tissue}_peaks_${suffix}
                    esac

                    for j in 0 1 2  #four sum stats datasets
                        do
                            varset=${gwas[$j]}
                            vars=${sumstats[$j]}
                            # echo $varset $vars 
                            echo $tt $tissue $suffix $peakset
                            qsub -cwd -b Y -N ${peakset}_${varset} -l h_vmem=50G "conda activate ldsc; python ~/ldsc/ldsc.py --h2 ${vars} --ref-ld-chr /illumina/scratch/deep_learning/asalcedo/baselineLD.,/illumina/scratch/deep_learning/asalcedo/scATAC/peaksets_processed/${peakset}_hg37. --frqfile-chr /illumina/scratch/deep_learning/asalcedo/1000G_Phase3_frq/1000G.EUR.QC. --w-ld-chr /illumina/scratch/deep_learning/asalcedo/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. --overlap-annot --print-coefficients --out brca_ss_vs_all_peaks_v2/${varset}_${peakset}"
                            
                        done
            done
    done