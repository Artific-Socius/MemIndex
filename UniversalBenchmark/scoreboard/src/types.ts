export interface WeightedBlock {
  passed: number;
  total: number;
  accuracy_percent: number;
}

export interface QuestionTypeBlock {
  passed: number;
  total: number;
  accuracy_percent: number;
}

export interface PerFileRow {
  path: string;
  basename: string;
  passed: number;
  total: number;
  accuracy_percent: number;
  source: string;
  dataset_name?: string;
  memory_type?: string;
  agent_model?: string;
  agent_identifier?: string;
}

export interface Dimensions {
  datasets: string[];
  memories: string[];
  agents: string[];
}

export interface GroupRow {
  dataset_name: string;
  memory_type: string;
  agent_model: string;
  passed: number;
  total: number;
  accuracy_percent: number;
}

export interface SummaryPayload {
  schema_version?: number;
  input_dir: string;
  name_regex: string;
  prefer_replay: boolean;
  recursive: boolean;
  skipped: Record<string, number>;
  file_count: number;
  weighted: WeightedBlock;
  by_question_type?: Record<string, QuestionTypeBlock>;
  dimensions?: Dimensions;
  by_group?: GroupRow[];
  per_file: PerFileRow[];
}
