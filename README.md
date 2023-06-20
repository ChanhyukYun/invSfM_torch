# invSfM_torch
Pytorch implementation of invSfM (Revealing scenes by inverting Structure-from-Motion reconstructions, Pittaluga et al., CVPR, 2019)   

Pretrained model weights files are available at https://github.com/francescopittaluga/invsfm   
Unzip 'wts.tar' with '/wts' directory remained at the repository   
   
   
If you use this code/model for your research, please cite the following paper:   
<pre>
<code>
@inproceedings{pittaluga2019revealing,
  title={Revealing scenes by inverting structure from motion reconstructions},
  author={Pittaluga, Francesco and Koppal, Sanjeev J and Bing Kang, Sing and Sinha, Sudipta N},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={145--154},
  year={2019}
}
</code>
</pre>   


# Paired-point lifting
Besides, we utilized this implementation codes for the brand-new privacy-preserving representation called 'Paired-point lifting (PPL)'.

More information about PPL can be found in https://github.com/Fusroda-h/ppl
<pre>
   <code>
   @inproceedings{lee2023paired,
     title={Paired-Point Lifting for Enhanced Privacy-Preserving Visual Localization},
     author={Lee, Chunghwan and Kim, Jaihoon and Yun, Chanhyuk and Hong, Je Hyeong},
     booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={17266--17275},
     year={2023}
   }
   </code>
</pre>

# Requirements
numpy==1.23.1

Pillow==9.2.0

scikit-image==0.19.3

scipy==1.9.0

torch==1.12.1

torchvision==0.13.1
