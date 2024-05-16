cd /media/raid/gongyu/projects/MVDRG/DGDR
python main_tta.py --root ../GDRBench_Data/ --algorithm GREEN --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_1view --output ../Result_4view/DG_MFIDDR_4viewss/GREEN
python main_tta.py --root ../GDRBench_Data/ --algorithm ERM --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_1view --output ../Result_4view/DG_MFIDDR_4viewss/ERM
python main_tta.py --root ../GDRBench_Data/ --algorithm MixStyleNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_1view --output ../Result_4view/DG_MFIDDR_4viewss/MixStyleNet
python main_tta.py --root ../GDRBench_Data/ --algorithm Fishr --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_1view --output ../Result_4view/DG_MFIDDR_4viewss/Fishr
python main_tta.py --root ../GDRBench_Data/ --algorithm MixupNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_1view --output ../Result_4view/DG_MFIDDR_4viewss/MixupNet
python main_tta.py --root ../GDRBench_Data/ --algorithm CABNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_1view --output ../Result_4view/DG_MFIDDR_4viewss/CABNet
python main_tta.py --root ../GDRBench_Data/ --algorithm GDRNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_1view --output ../Result_4view/DG_MFIDDR_4viewss/GDRNet

cd /media/raid/gongyu/projects/MVDRG/DGDR
python main_tta.py --root ../GDRBench_Data/ --algorithm GREEN --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_4views --output ../RetiGen/Result_4view/DG_MFIDDR_4viewss/GREEN
python main_tta.py --root ../GDRBench_Data/ --algorithm ERM --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_4views --output ../RetiGen/Result_4view/DG_MFIDDR_4viewss/ERM
python main_tta.py --root ../GDRBench_Data/ --algorithm MixStyleNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_4views --output ../RetiGen/Result_4view/DG_MFIDDR_4viewss/MixStyleNet
python main_tta.py --root ../GDRBench_Data/ --algorithm Fishr --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_4views --output ../RetiGen/Result_4view/DG_MFIDDR_4viewss/Fishr
python main_tta.py --root ../GDRBench_Data/ --algorithm MixupNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_4views --output ../RetiGen/Result_4view/DG_MFIDDR_4viewss/MixupNet
python main_tta.py --root ../GDRBench_Data/ --algorithm CABNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_4views --output ../RetiGen/Result_4view/DG_MFIDDR_4viewss/CABNet
python main_tta.py --root ../GDRBench_Data/ --algorithm GDRNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR DRTiD_1view --target-domains MFIDDR_4views --output ../RetiGen/Result_4view/DG_MFIDDR_4viewss/GDRNet

cd /media/raid/gongyu/projects/MVDRG/DGDR
python main_tta.py --root ../GDRBench_Data/ --algorithm GREEN --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_1view --output ../RetiGen/Result_2view/DG_DRTiD_2views/GREEN
python main_tta.py --root ../GDRBench_Data/ --algorithm ERM --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_1view --output ../RetiGen/Result_2view/DG_DRTiD_2views/ERM
python main_tta.py --root ../GDRBench_Data/ --algorithm MixStyleNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_1view --output ../RetiGen/Result_2view/DG_DRTiD_2views/MixStyleNet
python main_tta.py --root ../GDRBench_Data/ --algorithm Fishr --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_1view --output ../RetiGen/Result_2view/DG_DRTiD_2views/Fishr
python main_tta.py --root ../GDRBench_Data/ --algorithm MixupNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_1view --output ../RetiGen/Result_2view/DG_DRTiD_2views/MixupNet
python main_tta.py --root ../GDRBench_Data/ --algorithm CABNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_1view --output ../RetiGen/Result_2view/DG_DRTiD_2views/CABNet
python main_tta.py --root ../GDRBench_Data/ --algorithm GDRNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_1view --output ../RetiGen/Result_2view/DG_DRTiD_2views/GDRNet

cd /media/raid/gongyu/projects/MVDRG/DGDR
python main_tta.py --root ../GDRBench_Data/ --algorithm GREEN --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_2views --output ../RetiGen/Result_2view/DG_DRTiD_2views/GREEN
python main_tta.py --root ../GDRBench_Data/ --algorithm ERM --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_2views --output ../RetiGen/Result_2view/DG_DRTiD_2views/ERM
python main_tta.py --root ../GDRBench_Data/ --algorithm MixStyleNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_2views --output ../RetiGen/Result_2view/DG_DRTiD_2views/MixStyleNet
python main_tta.py --root ../GDRBench_Data/ --algorithm Fishr --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_2views --output ../RetiGen/Result_2view/DG_DRTiD_2views/Fishr
python main_tta.py --root ../GDRBench_Data/ --algorithm MixupNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_2views --output ../RetiGen/Result_2view/DG_DRTiD_2views/MixupNet
python main_tta.py --root ../GDRBench_Data/ --algorithm CABNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_2views --output ../RetiGen/Result_2view/DG_DRTiD_2views/CABNet
python main_tta.py --root ../GDRBench_Data/ --algorithm GDRNet --dg_mode DG --source-domains APTOS IDRID DEEPDR RLDR MFIDDR_1view --target-domains DRTiD_2views --output ../RetiGen/Result_2view/DG_DRTiD_2views/GDRNet