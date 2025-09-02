import numpy as np
import librosa
from pathlib import Path
from AWARE.utils import logger
from AWARE.utils.models import *
from AWARE.service import embed_watermark, detect_watermark
from attacks import PCMBitDepthConversion, MP3Compression, DeleteSamples, TimeStretch, Resample, RandomBandstop, SampleSupression, LowPassFilter, HighPassFilter, Cropout
from AWARE.metrics.audio import PESQ, SNR, BER, STOI

import logging
logger.setLevel(logging.DEBUG)

def main():

    attack_list = [ PCMBitDepthConversion(8), PCMBitDepthConversion(16), MP3Compression(9), MP3Compression(5), DeleteSamples(0.1),
                    DeleteSamples(0.25), DeleteSamples(0.5), DeleteSamples(0.75), TimeStretch(0.8), TimeStretch(0.9), TimeStretch(1.1), TimeStretch(1.2),
                    Resample(), RandomBandstop(), SampleSupression(0.1), SampleSupression(0.25), LowPassFilter() , HighPassFilter()] 

    print("Watermark Test Pipeline")
    print("=" * 50)
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cards_dir = project_root / "src" / "deltamark" / "cards"
    audio_folder_path = project_root / "common"
    print("gg")
    
    # Check if audio file exists
    if not audio_folder_path.exists():
        logger.error(f"Audio file path not found: {audio_folder_path}")
        logger.error("Please place correct folder path")
        return
    
    
    embedder, detector = load_model()

    watermark_length = 20
    watermark_bits = np.random.randint(0, 2, size=watermark_length, dtype=np.int32)

    pesq_metric = PESQ()
    stoi_metric = STOI()
    snr_metric = SNR()
    ber_metric = BER()    

    input_dir = Path(audio_folder_path)
    rec={}
    rec["pesq"] = []
    rec["stoi"] = []
    rec["orig"] = []

    for audio_file_path in input_dir.glob('*.*'):
        audio, sr = librosa.load(str(audio_file_path), sr=None, mono=True)
        print(sr)
        logger.info("Processing " + audio_file_path.name)

        '''
        if sr == 44100:
            from scipy.signal import resample_poly
            up, down = 1600, 4410
            audio = resample_poly(audio, up, down)
        '''
        sr=44100
        
        try:
            watermarked_audio = embed_watermark(audio, sample_rate=44100, watermark_bits = watermark_bits, model = embedder)
        except ValueError as e:
            # handle a specific exception
            print(f"Bad_input {audio_file_path.name}: ", e)
            continue

        detected_pattern = detect_watermark(watermarked_audio, sr, detector)
        
        ber_ = ber_metric(watermark_bits, detected_pattern)

        try:
            pesq_ = pesq_metric(watermarked_audio, audio, sr)
            rec["pesq"].append( pesq_ )
            logger.debug(f"PESQ : {pesq_}")
        except Exception as e:
            logger.info("Nema PESQ, Tisina")

        stoi_ = stoi_metric(watermarked_audio, audio, sr)
        if stoi_ > 0.1:
            rec["stoi"].append( stoi_ )
        
        rec["orig"].append( ber_ )
        
        logger.debug("orig: " + f"{ber_}")
        

        for attack in attack_list:
            name = attack.name
            
            wm_attacked = attack.apply(watermarked_audio, sr)
            
            detected_pattern = detect_watermark(wm_attacked, sr, detector)
            
            ber = ber_metric(watermark_bits, detected_pattern)
            
            if name not in rec:
                rec[name] = []
            rec[name].append(ber)

            if name in ['low_pass', 'high_pass', 'delete_0.1', 'delete_0.25', 'ts_0.9', 'ts_1.1']:
                logger.debug(name + ": " + f"{ber:.2f}")
            
    
    for att in rec.keys():
        items = np.array(rec[att])

        mean = items.mean()
        pop_std = items.std()

        logger.info(att + ": " + f"mean: {mean:.4f}" + f" pop_std: {pop_std:.4f}") 

if __name__ == "__main__":
    main()