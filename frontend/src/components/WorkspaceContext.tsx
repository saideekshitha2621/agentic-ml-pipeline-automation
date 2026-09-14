import { createContext, useContext, useState, type ReactNode } from "react";

interface WorkspaceState {
  datasetId: string | null;
  jobId: string | null;
  setDatasetId: (id: string | null) => void;
  setJobId: (id: string | null) => void;
}

const WorkspaceContext = createContext<WorkspaceState | undefined>(undefined);

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [datasetId, setDatasetId] = useState<string | null>(
    () => sessionStorage.getItem("datasetId"),
  );
  const [jobId, setJobId] = useState<string | null>(() => sessionStorage.getItem("jobId"));

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

  return (
    <WorkspaceContext.Provider
      value={{ datasetId, jobId, setDatasetId: setDatasetIdPersisted, setJobId: setJobIdPersisted }}
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
