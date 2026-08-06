import { useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

const STORAGE_PREFIX = "voxcpm360.";
// 打字與拖曳滑桿會讓值高頻變動，每次都同步寫 localStorage 會卡住主執行緒。
const WRITE_DEBOUNCE_MS = 250;

export function usePersistentState<T>(
  key: string,
  initialValue: T,
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    try {
      const saved = window.localStorage.getItem(`${STORAGE_PREFIX}${key}`);
      return saved === null ? initialValue : (JSON.parse(saved) as T);
    } catch {
      return initialValue;
    }
  });

  // 首次 render 的值就是剛從 storage 讀出來的（或預設值），寫回去沒有意義。
  const hydrated = useRef(false);

  useEffect(() => {
    if (!hydrated.current) {
      hydrated.current = true;
      return;
    }
    const timer = window.setTimeout(() => {
      try {
        window.localStorage.setItem(
          `${STORAGE_PREFIX}${key}`,
          JSON.stringify(value),
        );
      } catch {
        // Storage may be unavailable in private browsing; the UI still works in memory.
      }
    }, WRITE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [key, value]);

  return [value, setValue];
}
