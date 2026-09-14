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
