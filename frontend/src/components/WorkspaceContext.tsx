import { createContext, useContext, useState, type ReactNode } from "react";

interface WorkspaceState {
  datasetId: string | null;
  jobId: string | null;
  pipelineRunId: string | null;
  setDatasetId: (id: string | null) => void;
  setJobId: (id: string | null) => void;
  setPipelineRunId: (id: string | null) => void;
}

const WorkspaceContext = createContext<WorkspaceState | undefined>(undefined);

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [datasetId, setDatasetId] = useState<string | null>(
    () => sessionStorage.getItem("datasetId"),
  );
  const [jobId, setJobId] = useState<string | null>(() => sessionStorage.getItem("jobId"));
  const [pipelineRunId, setPipelineRunId] = useState<string | null>(
    () => sessionStorage.getItem("pipelineRunId"),
  );

  const setDatasetIdPersisted = (id: string | null) => {
    setDatasetId(id);
    if (id) sessionStorage.setItem("datasetId", id);
    else sessionStorage.removeItem("datasetId");
  };
  const setJobIdPersisted = (id: string | null) => {
    setJobId(id);
    if (id) sessionStorage.setItem("jobId", id);
    else sessionStorage.removeItem("jobId");
  };
  const setPipelineRunIdPersisted = (id: string | null) => {
    setPipelineRunId(id);
    if (id) sessionStorage.setItem("pipelineRunId", id);
    else sessionStorage.removeItem("pipelineRunId");
  };

  return (
    <WorkspaceContext.Provider
      value={{
        datasetId,
        jobId,
        pipelineRunId,
        setDatasetId: setDatasetIdPersisted,
        setJobId: setJobIdPersisted,
        setPipelineRunId: setPipelineRunIdPersisted,
      }}
    >
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace() {
  const ctx = useContext(WorkspaceContext);
  if (!ctx) throw new Error("useWorkspace must be used within WorkspaceProvider");
  return ctx;
}
