source SHORTCUTS
echo "with correction"
CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS=0 bash $MULTI_SCRIPT --purge true --retrain false --slow true
CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS=0 bash $MULTI_SCRIPT --purge true --retrain true --slow true

echo "without correction"
CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS=1 bash $MULTI_SCRIPT --retrain true --purge true --slow true
CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS=1 bash $MULTI_SCRIPT --retrain false --purge true --slow true