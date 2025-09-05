from pathlib import Path
from aware.embedding import AWAREEmbedder
from aware.detection import AWAREDetector
from aware.utils import load_config, logger

def load():
    script_dir = Path(__file__).parent.parent.parent
    cards_dir = script_dir / "cards"

    # Load configurations
    logger.info("Loading configurations...")
    try:
        config = load_config(cards_dir / "config.yaml")
        logger.info("Config loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configs: {e}")
        return


    logger.info("Creating embedder...")
    try:
        embedder = AWAREEmbedder(
            frame_length=config.get("frame_length", 1024),
            hop_length=config.get("hop_length", 256),
            window=config.get("window", "hann"),
            win_length=config.get("win_length", 1024),
            pattern_mode=config.get("pattern_mode", "bits2bipolar"),
            embedding_bands=tuple(config.get("embedding_bands", [500, 4000])),
            tolerance_db=config.get("tolerance_db", 6.0),
            num_iterations=config.get("num_iterations", 400),
            detection_net_cfg=config.get("detection_net_cfg", {}),
            optimizer_cfg=config.get("optimizer_cfg", {"name": "nadam", "params": {"lr": 0.1}}),
            scheduler_cfg=config.get("scheduler_cfg", {"name": "reduce_lr_on_plateau", "params": {"factor": 0.9, "patience": 500}}),
            loss=config.get("loss", "push_extremes"),
            verbose=config.get("verbose", True)
        )
        logger.info("Embedder created successfully")
        logger.info(f"   - Tolerance: {embedder.tolerance_db} dB")
        logger.info(f"   - Iterations: {embedder.num_iterations}")
        logger.info(f"   - Embedding bands: {embedder.embedding_bands} Hz")
        logger.info(f"   - Loss function: {embedder.loss}")   
        logger.info(f"   - Pattern mode: {embedder.pattern_mode}")
        logger.info(f"   - Optimizer: {embedder.optimizer_name}")
        logger.info(f"   - Scheduler: {embedder.scheduler_name}")
    except Exception as e:
        logger.error(f"Error creating embedder: {e}")
        import traceback
        traceback.print_exc()
        return
    


    logger.info("Creating detector...")
    try:
        detector = AWAREDetector(
            model=embedder.detection_net,
            threshold=config.get("threshold", 0.0),
            frame_length=config.get("frame_length", 1024),
            hop_length=config.get("hop_length", 256),
            window=config.get("window", "hann"),
            win_length=config.get("win_length", 1024),
            pattern_mode=config.get("pattern_mode", "bipolar"),
            embedding_bands=tuple(config.get("embedding_bands", [500, 4000]))
        )
        logger.info("Detector created successfully")
        logger.info(f"   - Threshold: {detector.threshold}")       
        logger.info("Model info:")
        logger.info(detector.detection_net.get_model_info())    
    except Exception as e:
        logger.error(f"Error creating detector: {e}")
        import traceback
        traceback.print_exc()
        return


    return embedder, detector
