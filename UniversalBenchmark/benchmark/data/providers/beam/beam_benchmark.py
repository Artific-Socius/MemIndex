from __future__ import annotations

import json
import os
import re
import ast
import shutil
from pathlib import Path
from typing import Any, Iterable

from benchmark.interfaces.benchmark import Benchmark
from benchmark.interfaces.scene import Scene, ConversationTurn
from benchmark.interfaces.question import Question
from benchmark.interfaces.scoring import ScoringConfig
from benchmark.interfaces.evidence import EvidenceBundle


def convert_chats_pickle_to_json(data: list) -> list:
    json_object = []
    if hasattr(data, "tolist"): data = data.tolist()
    for index, batch in enumerate(data):
        if hasattr(batch, "tolist"): batch = batch.tolist()
        batch_number = index + 1
        turns = []
        per_batch_turns = {}
        per_batch_turns['batch_number'] = batch_number
        time_anchor = None
        single_turn = []
        for index2, message in enumerate(batch):
            if hasattr(message, "tolist"): message = message.tolist()
            if isinstance(message, list) and len(message) > 0: message = message[0]
            if not isinstance(message, dict): continue

            if "question_type" in message.keys() and message['question_type'] == "main_question" and single_turn != []:
                if "time_anchor" in message.keys():
                    time_anchor = message['time_anchor']
                turns.append(single_turn)
                single_turn = []
                single_turn.append(message)
            else:
                single_turn.append(message)

        if single_turn != []:
            turns.append(single_turn)
        per_batch_turns['turns'] = turns
        per_batch_turns['time_anchor'] = time_anchor
        json_object.append(per_batch_turns)

    return json_object

def combine_10m_chats(chats: list):
    new_chat = []
    last_id = 0
    for index, chat in enumerate(chats):
        if last_id != 0:
            for batch_number, batch in enumerate(chat):
                turns = batch['turns']
                for turn_number, turn in enumerate(turns):
                    for message_number, message in enumerate(turn):
                        chat[batch_number]['turns'][turn_number][message_number]['id'] += last_id + 1

        last_id = chat[-1]['turns'][-1][-1]["id"]
        new_chat.append({
            f"plan-{index+1}": chat
        })
    return new_chat

class BeamScene(Scene):
    def __init__(self, scene_id: str, scene_dir: Path, scale: str):
        self._scene_id = scene_id
        self._scene_dir = scene_dir
        self._scale = scale
        self._scene_name = f"BEAM {self._scale} - {self._scene_id}"

        chat_file = self._scene_dir / "chat.json"
        self._chat_data = []
        if chat_file.is_file():
            with open(chat_file, "r", encoding="utf-8") as f:
                self._chat_data = json.load(f)

        probing_file = self._scene_dir / "probing_questions" / "probing_questions.json"
        self._probing_data = {}
        if probing_file.is_file():
            with open(probing_file, "r", encoding="utf-8") as f:
                self._probing_data = json.load(f)

        topic_file = self._scene_dir / "topic.json"
        self._narrative = ""
        if topic_file.is_file():
            with open(topic_file, "r", encoding="utf-8") as f:
                self._narrative = str(json.load(f))

    @property
    def scene_id(self) -> str:
        return self._scene_id

    @property
    def scene_name(self) -> str:
        return self._scene_name

    @property
    def task_type(self) -> str:
        return "beam_conversation"

    def background_text(self, max_chars: int | None = None) -> str:
        if max_chars is not None:
            return self._narrative[:max_chars]
        return self._narrative

    def background_documents(self) -> list[str]:
        return []

    def conversation_history(self) -> list[ConversationTurn]:
        turns = []
        
        # chat_data might be a list of batches (100K, 500K, 1M) or a list of plans (10M)
        # We need to flatten the messages to create ConversationTurns
        
        def process_batches(batches):
            for batch in batches:
                batch_turns = batch.get("turns", [])
                for turn_pair in batch_turns:
                    user_msg = ""
                    asst_msg = ""
                    for msg in turn_pair:
                        role = str(msg.get("role", "")).strip().lower()
                        content = str(msg.get("content", ""))
                        if role == "user":
                            user_msg += content + "\n"
                        elif role == "assistant":
                            asst_msg += content + "\n"
                    if user_msg or asst_msg:
                        turns.append(ConversationTurn(user_message=user_msg.strip(), assistant_response=asst_msg.strip()))

        if self._scale == "10M":
            for plan_dict in self._chat_data:
                for plan_name, batches in plan_dict.items():
                    process_batches(batches)
        else:
            process_batches(self._chat_data)
            
        return turns

    def question_count(self) -> int:
        count = 0
        for category_questions in self._probing_data.values():
            count += len(category_questions)
        return count

    def get_question_by_id(self, question_id: str) -> Question:
        for q in self.questions():
            if q.question_id == question_id:
                return q
        raise KeyError(f"Question {question_id} not found in {self._scene_id}")

    def questions(self) -> Iterable[Question]:
        q_counter = 0
        for ability, category_questions in self._probing_data.items():
            for idx, q_data in enumerate(category_questions):
                gt = q_data.get("ideal_response") or q_data.get("ideal_answer") or q_data.get("answer") or ""
                qid = str(q_counter)
                
                payload = dict(q_data)
                payload["memory_ability"] = ability
                
                yield Question(
                    question_id=qid,
                    question_text=q_data.get("question", ""),
                    ground_truth=gt,
                    evidence=EvidenceBundle(
                        evidence_type="beam_chat",
                        payload=payload,
                        allow_missing_references=True
                    ),
                    scoring=ScoringConfig(
                        eval_mode="llm",
                        eval_prompt_key="[TODO]",
                        max_score=1.0
                    )
                )
                q_counter += 1


class BeamBenchmark(Benchmark):
    def __init__(self, scale: str):
        self._scale = scale
        base_raw = Path(__file__).resolve().parents[2] / "raw" / "Mohammadta"
        
        self._decoded_root = base_raw / ".decoded" / scale
        
        if scale == "10M":
            self._parquet_dir = base_raw / "BEAM-10M" / "data"
        else:
            self._parquet_dir = base_raw / "BEAM" / "data"

    def _ensure_decoded(self):
        if self._decoded_root.is_dir() and any(self._decoded_root.iterdir()):
            return # Already decoded
            
        if not self._parquet_dir.is_dir():
            return # No data to decode
            
        import pandas as pd
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                return super().default(obj)

        parquet_files = list(self._parquet_dir.glob(f"{self._scale}-*.parquet"))
        if not parquet_files:
            return
            
        print(f"[BeamBenchmark] Loading {self._scale} parquets...")
        df = pd.concat([pd.read_parquet(p) for p in parquet_files])
        
        os.makedirs(self._decoded_root, exist_ok=True)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Decoding {self._scale} BEAM Data"):
            conversation_id = str(row['conversation_id'])
            topic = row['conversation_seed']
            chat = row['chat']
            probing_questions = row['probing_questions']
            
            conversation_dir = self._decoded_root / conversation_id
            os.makedirs(conversation_dir, exist_ok=True)
            
            if self._scale == "10M":
                plans = row['plans']
                if hasattr(plans, "tolist"): plans = plans.tolist()
                chats = []
                for plan_id, plan in enumerate(plans):
                    if hasattr(plan, "tolist"): plan = plan.tolist()
                    if isinstance(plan, list) and len(plan) > 0: plan = plan[0]
                    p_chat = plan.get("chat", []) if isinstance(plan, dict) else []
                    json_chat = convert_chats_pickle_to_json(data=p_chat)
                    chats.append(json_chat)
                json_chat = combine_10m_chats(chats=chats)
            else:
                json_chat = convert_chats_pickle_to_json(data=chat)
                
            chat_dir = conversation_dir / "chat.json"
            with open(chat_dir, "w", encoding="utf-8") as f:
                json.dump(json_chat, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                
            topic_dir = conversation_dir / "topic.json"
            with open(topic_dir, "w", encoding="utf-8") as f:
                json.dump(topic, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                
            probing_questions_dir = conversation_dir / "probing_questions"
            os.makedirs(probing_questions_dir, exist_ok=True)
            probing_questions_file = probing_questions_dir / "probing_questions.json"
            
            if isinstance(probing_questions, str):
                try:
                    probing_questions_data = json.loads(probing_questions)
                except json.JSONDecodeError:
                    probing_questions_data = ast.literal_eval(probing_questions)
            else:
                probing_questions_data = probing_questions
                
            with open(probing_questions_file, "w", encoding="utf-8") as f:
                json.dump(probing_questions_data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    @property
    def benchmark_name(self) -> str:
        return f"Mohammadta/BEAM:{self._scale}"

    @property
    def raw_root(self) -> Path:
        return self._decoded_root

    @property
    def eval_prompt(self) -> str | None:
        return "[TODO]"

    def list_scenes(self) -> Iterable[str]:
        self._ensure_decoded()
        if not self._decoded_root.is_dir():
            return
            
        for p in self._decoded_root.iterdir():
            if p.is_dir():
                yield p.name

    def get_scene(self, scene_id: str) -> Scene:
        self._ensure_decoded()
        scene_dir = self._decoded_root / scene_id
        if not scene_dir.is_dir():
            raise KeyError(f"Scene dir not found: {scene_dir}")
            
        return BeamScene(scene_id, scene_dir, self._scale)
