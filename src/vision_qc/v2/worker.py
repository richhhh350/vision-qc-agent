import argparse
import json
from .feedback import retrain_feedback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run","state","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--steps",type=int,default=120)
    args = parser.parse_args()
    print(json.dumps(retrain_feedback(args.run,args.state,args.output,args.steps),indent=2),flush=True)
