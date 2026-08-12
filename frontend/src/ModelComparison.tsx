import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  CircleAlert,
  Info,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchModelRegistry } from "./api";
import { usePersistentState } from "./usePersistentState";
import type {
  ModelRegistry,
  ModelRegistryEntry,
} from "./types";

type NumericSortKey = "val_loss" | "best_epoch" | "size_gb";
type SortKey = "name" | "family_method" | NumericSortKey;
type SortDirection = "asc" | "desc";
type ModelFamily = "native" | "barbet";

const EMPTY_REGISTRY: ModelRegistry = { _val_sets: {}, models: [] };
const METHOD_LABELS: Record<string, string> = {
  "full-finetune": "全參微調",
  lora: "LoRA",
  "bluemagpie-pretrained": "預訓練／合併",
  "bluemagpie-tslm": "TSLM 訓練",
  "bluemagpie-tslm-avg": "權重平均",
  "bluemagpie-full": "Full 訓練",
  "bluemagpie-bridge": "Bridge 訓練",
};

const FAMILY_LABELS: Record<ModelFamily, string> = {
  native: "原生 VoxCPM2",
  barbet: "Barbet／BlueMagpie",
};

const FAMILY_DESCRIPTIONS: Record<ModelFamily, string> = {
  native:
    "使用 VoxCPM2 原生 MiniCPM4 TSLM；全參微調與 LoRA 是同一架構下的不同訓練方式。",
  barbet:
    "以 Barbet 取代原生 TSLM；Bridge 只訓練對齊層，TSLM 另訓練 Barbet 與語者投影層。",
};

function getModelFamily(model: ModelRegistryEntry): ModelFamily {
  return model.method.startsWith("bluemagpie-") ? "barbet" : "native";
}

const METHOD_DESCRIPTIONS: Record<string, string> = {
  "bluemagpie-pretrained": "未經本專案訓練的官方基礎模型，或由既有權重合併而成的模型。",
  "bluemagpie-bridge":
    "只訓練橋接層（約 1.5% 參數），用來對齊 Barbet 與 VoxCPM2 的隱藏表示空間。",
  "bluemagpie-tslm":
    "訓練橋接層、Barbet TSLM 與語者投影層，VoxCPM2 聲學主幹維持凍結。",
  "bluemagpie-tslm-avg": "由同一輪訓練的多個 checkpoint 進行權重平均。",
  "bluemagpie-full": "除 AudioVAE 外解凍完整 BlueMagpie 模型進行訓練。",
};

function getMethodLabel(method: string): string {
  return METHOD_LABELS[method] || method;
}

function isFiniteNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function compareOptionalNumbers(
  left: number | undefined,
  right: number | undefined,
  direction: SortDirection,
): number {
  const leftIsNumber = isFiniteNumber(left);
  const rightIsNumber = isFiniteNumber(right);
  if (!leftIsNumber || !rightIsNumber) {
    if (leftIsNumber === rightIsNumber) {
      return 0;
    }
    // Missing metrics stay at the bottom for both ascending and descending sorts.
    return leftIsNumber ? -1 : 1;
  }
  return direction === "asc" ? left - right : right - left;
}

function compareModels(
  left: ModelRegistryEntry,
  right: ModelRegistryEntry,
  sortKey: SortKey,
  direction: SortDirection,
): number {
  const directionFactor = direction === "asc" ? 1 : -1;
  if (sortKey === "name") {
    return left.name.localeCompare(right.name, "zh-TW") * directionFactor;
  }

  if (sortKey === "family_method") {
    const familyDifference = FAMILY_LABELS[getModelFamily(left)].localeCompare(
      FAMILY_LABELS[getModelFamily(right)],
      "zh-TW",
    );
    const methodDifference = getMethodLabel(left.method).localeCompare(
      getMethodLabel(right.method),
      "zh-TW",
    );
    return (familyDifference || methodDifference) * directionFactor;
  }

  return compareOptionalNumbers(left[sortKey], right[sortKey], direction);
}

function formatMetric(value: number | undefined, digits: number): string {
  return isFiniteNumber(value) ? value.toFixed(digits) : "—";
}

function formatStep(step: number | undefined): string {
  return isFiniteNumber(step) ? step.toLocaleString("zh-TW") : "—";
}

function formatSize(sizeGb: number | undefined): string {
  if (!isFiniteNumber(sizeGb)) {
    return "—";
  }
  if (sizeGb < 1) {
    return `${Math.round(sizeGb * 1000).toLocaleString("zh-TW")} MB`;
  }
  return `${sizeGb.toLocaleString("zh-TW", {
    maximumFractionDigits: 2,
  })} GB`;
}

function uniqueValues(models: ModelRegistryEntry[], key: keyof ModelRegistryEntry) {
  return [...new Set(models.map((model) => String(model[key])))].sort((a, b) =>
    a.localeCompare(b, "zh-TW"),
  );
}

interface SortButtonProps {
  activeKey: SortKey;
  direction: SortDirection;
  field: SortKey;
  label: string;
  onSort: (field: SortKey) => void;
}

function SortButton({
  activeKey,
  direction,
  field,
  label,
  onSort,
}: SortButtonProps) {
  const active = activeKey === field;
  const Icon = active ? (direction === "asc" ? ArrowUp : ArrowDown) : ArrowUpDown;
  return (
    <button
      type="button"
      className={`table-sort ${active ? "active" : ""}`}
      onClick={() => onSort(field)}
      title={`依${label}排序`}
    >
      {label}
      <Icon size={13} />
    </button>
  );
}

interface ModelComparisonProps {
  reloadToken: number;
}

export default function ModelComparison({ reloadToken }: ModelComparisonProps) {
  const [registry, setRegistry] = useState<ModelRegistry>(EMPTY_REGISTRY);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [familyFilter, setFamilyFilter] = usePersistentState(
    "comparison.family",
    "",
  );
  const [methodFilter, setMethodFilter] = usePersistentState(
    "comparison.method",
    "",
  );
  const [trainDataFilter, setTrainDataFilter] = usePersistentState(
    "comparison.train-data",
    "",
  );
  const [valSetFilter, setValSetFilter] = usePersistentState(
    "comparison.val-set",
    "",
  );
  const [sortKey, setSortKey] = usePersistentState<SortKey>(
    "comparison.sort-key",
    "val_loss",
  );
  const [sortDirection, setSortDirection] =
    usePersistentState<SortDirection>("comparison.sort-direction", "asc");

  const loadRegistry = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError("");
    try {
      setRegistry(await fetchModelRegistry(signal));
    } catch (loadError) {
      if (loadError instanceof DOMException && loadError.name === "AbortError") {
        return;
      }
      setError(loadError instanceof Error ? loadError.message : "無法取得模型比較資料");
    } finally {
      if (!signal?.aborted) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadRegistry(controller.signal);
    return () => controller.abort();
  }, [loadRegistry, reloadToken]);

  const methods = useMemo(
    () => uniqueValues(registry.models, "method"),
    [registry.models],
  );
  const trainDataSets = useMemo(
    () => uniqueValues(registry.models, "train_data"),
    [registry.models],
  );
  const validationSets = useMemo(
    () => uniqueValues(registry.models, "val_set"),
    [registry.models],
  );

  const filteredModels = useMemo(
    () =>
      registry.models.filter(
        (model) =>
          (!familyFilter || getModelFamily(model) === familyFilter) &&
          (!methodFilter || model.method === methodFilter) &&
          (!trainDataFilter || model.train_data === trainDataFilter) &&
          (!valSetFilter || model.val_set === valSetFilter),
      ),
    [registry.models, familyFilter, methodFilter, trainDataFilter, valSetFilter],
  );

  const groups = useMemo(() => {
    const registryOrder = Object.keys(registry._val_sets);
    const groupNames = [
      ...registryOrder,
      ...validationSets.filter((name) => !registryOrder.includes(name)),
    ];
    return groupNames
      .map((valSet) => ({
        valSet,
        description: registry._val_sets[valSet] || "此組模型使用相同驗證集。",
        models: filteredModels
          .filter((model) => model.val_set === valSet)
          .sort((left, right) => {
            const difference = compareModels(
              left,
              right,
              sortKey,
              sortDirection,
            );
            if (difference !== 0) {
              return difference;
            }
            return left.name.localeCompare(right.name, "zh-TW");
          }),
      }))
      .filter((group) => group.models.length > 0);
  }, [filteredModels, registry._val_sets, sortDirection, sortKey, validationSets]);

  const handleSort = (field: SortKey) => {
    if (sortKey === field) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(field);
    setSortDirection("asc");
  };

  const resetFilters = () => {
    setFamilyFilter("");
    setMethodFilter("");
    setTrainDataFilter("");
    setValSetFilter("");
  };

  return (
    <section className="comparison-view" aria-label="模型訓練成果">
      <div className="comparison-warning" role="note">
        <Info size={19} />
        <p>
          <strong>Val loss 只能在同一個驗證集內比較，也不等同實際聽感。</strong>
          naer 的 0.93 不代表優於 tai8 的 1.09；未評估或刻意取後段 epoch
          的模型會以「—」顯示，排序時置於該組末尾。
        </p>
      </div>

      <div className="family-explainer" aria-label="模型家族差異">
        {(["native", "barbet"] as const).map((family) => (
          <div key={family}>
            <span className="family-badge" data-family={family}>
              {FAMILY_LABELS[family]}
            </span>
            <p>{FAMILY_DESCRIPTIONS[family]}</p>
          </div>
        ))}
      </div>

      <div className="registry-filters panel" aria-label="模型篩選">
        <label>
          <span className="field-label">模型家族</span>
          <span className="select-wrap">
            <select value={familyFilter} onChange={(event) => setFamilyFilter(event.target.value)}>
              <option value="">全部家族</option>
              <option value="native">{FAMILY_LABELS.native}</option>
              <option value="barbet">{FAMILY_LABELS.barbet}</option>
            </select>
          </span>
        </label>
        <label>
          <span className="field-label">訓練方法</span>
          <span className="select-wrap">
            <select value={methodFilter} onChange={(event) => setMethodFilter(event.target.value)}>
              <option value="">全部方法</option>
              {methods.map((method) => (
                <option key={method} value={method}>
                  {getMethodLabel(method)}
                </option>
              ))}
            </select>
          </span>
        </label>
        <label>
          <span className="field-label">訓練資料</span>
          <span className="select-wrap">
            <select
              value={trainDataFilter}
              onChange={(event) => setTrainDataFilter(event.target.value)}
            >
              <option value="">全部資料</option>
              {trainDataSets.map((dataSet) => (
                <option key={dataSet} value={dataSet}>
                  {dataSet}
                </option>
              ))}
            </select>
          </span>
        </label>
        <label>
          <span className="field-label">驗證集</span>
          <span className="select-wrap">
            <select value={valSetFilter} onChange={(event) => setValSetFilter(event.target.value)}>
              <option value="">全部驗證集</option>
              {validationSets.map((valSet) => (
                <option key={valSet} value={valSet}>
                  {valSet}
                </option>
              ))}
            </select>
          </span>
        </label>
        <button type="button" className="filter-reset" onClick={resetFilters}>
          <RotateCcw size={15} />
          清除篩選
        </button>
      </div>

      {error && (
        <div className="alert error-alert registry-error">
          <CircleAlert size={18} />
          <span>{error}</span>
        </div>
      )}

      {loading && registry.models.length === 0 ? (
        <div className="registry-loading panel">
          <LoaderCircle size={22} className="spin" />
          正在載入模型資料…
        </div>
      ) : groups.length === 0 ? (
        <div className="registry-empty panel">
          <h3>沒有符合條件的模型</h3>
          <p>調整篩選條件後再試一次。</p>
        </div>
      ) : (
        <div className="registry-groups">
          {groups.map((group) => (
            <section className="registry-group panel" key={group.valSet}>
              <header className="registry-group-heading">
                <div>
                  <span className="val-set-badge">
                    {group.valSet === "-" ? "VAL · 未評估" : `VAL · ${group.valSet}`}
                  </span>
                  <h3>
                    {group.valSet === "-"
                      ? "未提供驗證結果"
                      : `${group.valSet} 驗證集`}
                  </h3>
                  <p>
                    {group.valSet === "-"
                      ? "基礎或合併模型，未提供可供此表比較的驗證指標。"
                      : group.description}
                  </p>
                </div>
                <span>{group.models.length} 個模型</span>
              </header>

              <div className="registry-table-wrap">
                <table className="registry-table">
                  <thead>
                    <tr>
                      <th scope="col">
                        <SortButton
                          activeKey={sortKey}
                          direction={sortDirection}
                          field="name"
                          label="名稱"
                          onSort={handleSort}
                        />
                      </th>
                      <th scope="col">
                        <SortButton
                          activeKey={sortKey}
                          direction={sortDirection}
                          field="family_method"
                          label="架構／方法"
                          onSort={handleSort}
                        />
                      </th>
                      <th scope="col">訓練資料</th>
                      <th scope="col">
                        <SortButton
                          activeKey={sortKey}
                          direction={sortDirection}
                          field="val_loss"
                          label="Val loss"
                          onSort={handleSort}
                        />
                      </th>
                      <th scope="col">
                        <SortButton
                          activeKey={sortKey}
                          direction={sortDirection}
                          field="best_epoch"
                          label="Epoch"
                          onSort={handleSort}
                        />
                      </th>
                      <th scope="col">Step</th>
                      <th scope="col">LR／Batch</th>
                      <th scope="col">
                        <SortButton
                          activeKey={sortKey}
                          direction={sortDirection}
                          field="size_gb"
                          label="大小"
                          onSort={handleSort}
                        />
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {group.models.map((model) => (
                      <tr key={model.name}>
                        <td>
                          <strong className="registry-model-name">{model.name}</strong>
                          {model.note && (
                            <small className="registry-model-note" title={model.note}>
                              {model.note}
                            </small>
                          )}
                        </td>
                        <td>
                          <span
                            className="family-badge"
                            data-family={getModelFamily(model)}
                          >
                            {FAMILY_LABELS[getModelFamily(model)]}
                          </span>
                          <small
                            className="registry-arch"
                            title={METHOD_DESCRIPTIONS[model.method]}
                          >
                            {getMethodLabel(model.method)} · {model.arch.match(/\((.+)\)/)?.[1] || model.arch}
                          </small>
                        </td>
                        <td>
                          <span className="data-set-label">{model.train_data}</span>
                        </td>
                        <td className="metric-cell primary-metric">
                          {formatMetric(model.val_loss, 4)}
                        </td>
                        <td className="metric-cell">
                          {formatMetric(model.best_epoch, 2)}
                        </td>
                        <td className="metric-cell">{formatStep(model.best_step)}</td>
                        <td>
                          {model.lr ||
                          model.effective_batch !== undefined ||
                          model.lora_r !== undefined ? (
                            <>
                              <span className="training-detail">
                                {model.lr ? `LR ${model.lr}` : "LR —"}
                              </span>
                              <small>
                                {model.effective_batch !== undefined
                                  ? `Batch ${model.effective_batch}`
                                  : "Batch —"}
                                {model.lora_r ? ` · r${model.lora_r}` : ""}
                              </small>
                            </>
                          ) : (
                            <span className="training-detail">—</span>
                          )}
                        </td>
                        <td className="metric-cell size-cell">{formatSize(model.size_gb)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ))}
        </div>
      )}
    </section>
  );
}
