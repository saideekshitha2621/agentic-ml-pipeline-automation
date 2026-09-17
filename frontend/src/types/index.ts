export interface Dataset {
  id: string;
  filename: string;
  uploaded_at: string;
  n_rows: number;
  n_columns: number;
  data_quality_score: number;
}

export interface MissingValueRow {
  column: string;
  dtype: string;
  missing_count: number;
  missing_pct: number;
  n_unique: number;
}

export interface OutlierRow {
  column: string;
  method: string;
  n_outliers: number;
  pct_outliers: number;
  lower_bound: number | null;
  upper_bound: number | null;
}

export interface CategorySuggestion {
  column: string;
  issue: string;
  examples: string[];
  suggestion: string;
}

export interface DataProfile {
  n_rows: number;
  n_columns: number;
  n_duplicates: number;
  data_quality_score: number;
  missing_values: MissingValueRow[];
  outliers: OutlierRow[];
  category_suggestions: CategorySuggestion[];
}

export interface PreprocessingPlan {
  id: string;
  dataset_id: string;
  numerical_columns: string[];
  categorical_columns: string[];
  dropped_columns: string[];
  numerical_impute_strategy: string;
  categorical_impute_strategy: string;
  scaling_method: string;
  drop_duplicates: boolean;
  pca_enabled: boolean;
  pca_variance_target: number;
}

export interface PreprocessingPlanUpdate {
  numerical_columns: string[];
  categorical_columns: string[];
  dropped_columns: string[];
  numerical_impute_strategy: string;
  categorical_impute_strategy: string;
  scaling_method: string;
  drop_duplicates: boolean;
  pca_enabled: boolean;
  pca_variance_target: number;
}

export interface PreprocessingReport {
  initial_shape: number[];
  final_shape: number[];
  duplicates_found: number;
  duplicates_removed: number;
  imputation: Record<string, unknown>;
  scaling: Record<string, unknown>;
}

export interface PCAPreview {
  explained_variance_ratio: number[];
  cumulative_variance: number[];
  components: string[];
}

export interface PCAReport {
  n_components: number;
  explained_variance_ratio: number[];
  cumulative_explained_variance: number[];
  total_variance_retained: number;
  original_n_features: number;
}

export interface Job {
  id: string;
  dataset_id: string;
  preprocessing_plan_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress_pct: number;
  log_lines: string[];
  pca_report_json: Record<string, unknown>;
  started_at: string;
  completed_at: string | null;
  error_message: string | null;
}

export interface ClusterRun {
  id: string;
  job_id: string;
  algorithm: string;
  params_json: Record<string, unknown>;
  extra_json: Record<string, unknown>;
  n_clusters: number;
  n_noise: number;
  noise_pct: number;
  silhouette_score: number | null;
  davies_bouldin_score: number | null;
  calinski_harabasz_score: number | null;
  composite_score: number | null;
  rank: number | null;
}

export interface Recommendation {
  cluster_run: ClusterRun;
  rationale: string;
  strengths: string[];
  weaknesses: string[];
  cluster_size_breakdown: Record<string, number>;
}

export interface Approval {
  id: string;
  job_id: string;
  cluster_run_id: string;
  approved_by: string;
  approved_at: string;
  notes: string | null;
}

export interface ScatterPoint {
  x: number;
  y: number;
  cluster: string;
}

export interface ClusterSizeRow {
  cluster: string;
  count: number;
  pct_of_total: number;
}

export interface MetricComparisonRow {
  run_id: string;
  algorithm: string;
  silhouette_score: number | null;
  davies_bouldin_score: number | null;
  calinski_harabasz_score: number | null;
}

export interface VisualizationBundle {
  scatter: ScatterPoint[];
  cluster_sizes: ClusterSizeRow[];
  metric_comparison: MetricComparisonRow[];
}

export interface ClusterInterpretation {
  profiles: Record<string, unknown>[];
  feature_importance: Record<string, unknown>[];
  summaries: Record<string, string>;
  suggested_names: Record<string, string>;
}

export type PipelineRunStatus =
  | "profiling"
  | "awaiting_problem_approval"
  | "data_validation"
  | "awaiting_validation_approval"
  | "cleaning_plan"
  | "awaiting_cleaning_approval"
  | "transformation"
  | "awaiting_transformation_approval"
  | "train_test_split"
  | "awaiting_split_approval"
  | "algorithm_recommendation"
  | "awaiting_algorithm_approval"
  | "training"
  | "hyperparameter_optimization"
  | "evaluation"
  | "awaiting_recommendation_approval"
  | "reporting"
  | "completed"
  | "failed";

export const PIPELINE_STAGES: { status: PipelineRunStatus; gate?: PipelineRunStatus; label: string; agentName: string }[] = [
  { status: "profiling", label: "Dataset Understanding & Problem Detection", agentName: "problem_detection" },
  { status: "data_validation", gate: "awaiting_validation_approval", label: "Data Validation", agentName: "data_validation" },
  { status: "cleaning_plan", gate: "awaiting_cleaning_approval", label: "Data Cleaning Plan", agentName: "cleaning_plan" },
  { status: "transformation", gate: "awaiting_transformation_approval", label: "Data Transformation", agentName: "transformation" },
  { status: "train_test_split", gate: "awaiting_split_approval", label: "Train/Test Split", agentName: "train_test_split" },
  { status: "algorithm_recommendation", gate: "awaiting_algorithm_approval", label: "Algorithm Recommendation", agentName: "algorithm_recommendation" },
  { status: "training", label: "Training", agentName: "model_selection" },
  { status: "hyperparameter_optimization", label: "Hyperparameter Optimization", agentName: "hyperparameter_optimization" },
  { status: "evaluation", label: "Evaluation", agentName: "evaluation" },
  { status: "awaiting_recommendation_approval", label: "Recommendation", agentName: "recommendation" },
  { status: "reporting", label: "Final Report", agentName: "reporting" },
];

export interface PipelineRun {
  id: string;
  dataset_id: string;
  declared_target: string | null;
  status: PipelineRunStatus;
  problem_type: string | null;
  job_id: string | null;
  created_at: string;
  updated_at: string;
  error_message: string | null;
  champion_model_path: string | null;
  feature_schema_json: Record<string, { type: "numeric" | "categorical"; min?: number; max?: number; options?: string[]; default: unknown }>;
  feature_importance_json: { features?: { feature: string; importance: number; importance_pct: number }[]; target_classes?: string[] };
}

export interface ChatMessage {
  id: string;
  pipeline_run_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface TrainingProgress {
  algorithms: { algorithm: string; status: "running" | "completed" | "failed" }[];
  job_status: string | null;
  progress_pct?: number;
}

export type AgentDecisionStatus = "proposed" | "approved" | "edited" | "rejected";

export interface AgentDecision {
  id: string;
  pipeline_run_id: string;
  agent_name: string;
  stage: string;
  decision_json: Record<string, unknown>;
  confidence: number | null;
  reasoning_text: string;
  status: AgentDecisionStatus;
  human_edits_json: Record<string, unknown> | null;
  override_reason: string | null;
  approved_by: string | null;
  approved_at: string | null;
  created_at: string;
}

export interface RecommendationCandidate {
  cluster_run_id: string;
  algorithm: string;
  params: Record<string, unknown>;
  rank: number;
  composite_score: number | null;
  rationale: string;
  strengths: string[];
  weaknesses: string[];
  cluster_size_breakdown: Record<string, number>;
  business_benefits?: string[];
  why_not_chosen?: string;
}

export interface ExecutiveSummary {
  business_problem: {
    statement: string;
    prediction_objective: string;
    key_features: string[];
    business_value: string;
  } | null;
  dataset_overview: { n_rows: number; n_columns: number; quality_summary: string } | null;
  ml_type: string | null;
  target_variable: string | null;
  recommended_model: { algorithm: string; rationale: string; business_benefits: string[] } | null;
  performance_summary: { headline_metric_sentence: string | null; confidence: string | null } | null;
}

export interface AgentRecommendation {
  top_choice: RecommendationCandidate;
  alternatives: RecommendationCandidate[];
  confidence: "high" | "low";
}

export interface PipelineReport {
  title: string;
  executive_summary: {
    problem_statement: string | null;
    dataset_overview: {
      dataset_name: string;
      total_records: number;
      features_analyzed: string[];
      target_variable: string | null;
      business_domain: string;
    };
  };
  data_quality_summary: {
    issues_detected: { missing_values: number; duplicate_records: number; invalid_data: number };
    actions_taken: { column: string; missing_values: number; action: string; reason: string }[];
    duplicate_handling: string;
  };
  data_preparation_summary: {
    steps: string[];
    train_pct: number | null;
    test_pct: number | null;
  };
  model_selection_summary: {
    algorithms_evaluated: string[];
    selected_model: string | null;
    why_selected: string | null;
    hyperparameter_optimization: string;
  };
  model_performance: {
    reliability_sentence: string | null;
    confidence_level: string;
    confidence_explanation: string;
  };
  key_insights: string[];
  prediction_capability: {
    description: string | null;
    example_predictions: {
      input: Record<string, unknown>;
      prediction: string | number;
      confidence: string;
      suggested_business_action: string | null;
      created_at: string;
    }[];
  };
  business_recommendations: string[];
  conclusion: string;
  technical_appendix: {
    model_details: Record<string, unknown>;
    hyperparameters: Record<string, unknown>;
    train_test_split: { training: number; testing: number } | null;
    model_training_summary: {
      algorithms_evaluated: number;
      hyperparameter_optimization: boolean;
      champion_model: string | null;
      prediction_type: string;
    };
  };
  final_outcome: string;
}
