import { useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const socHelp =
  "배터리에 남아 있는 전기의 비율입니다. SOC 50%는 사용할 수 있는 전체 저장 용량의 절반이 남았다는 뜻입니다.";
const supplyHelp =
  "해당 시간에 전력망으로 공급하기로 계획한 전력량입니다. MWh는 전기의 양을 나타내는 단위입니다.";
const descriptions: Record<string, string> = {
  "선로 조건":
    "태양광과 ESS가 전력망으로 내보낼 수 있는 최대 출력 조건입니다. 이 화면에서는 이미 계산된 조건별 결과를 비교합니다.",
  설비: "ESS는 전기를 저장했다가 필요할 때 내보내는 에너지저장장치입니다. 후보마다 출력·저장 용량·비용 가정이 다릅니다.",
  "ESS 출력":
    "한 번에 충전하거나 방전할 수 있는 전력의 크기입니다. MW 값이 클수록 같은 시간에 더 많은 전기를 주고받을 수 있습니다.",
  "저장 용량":
    "배터리에 담을 수 있는 전기의 양입니다. MWh는 저장량, MW는 충전·방전 속도를 나타냅니다.",
  "연간 VPP 수입":
    "발전·저장 설비를 묶어 운영하는 VPP 사업자에게 배분되는 연간 수입입니다. 투자·유지·교체비를 빼기 전 금액입니다.",
  "VPP 귀속 수입":
    "전체 정산액 중 VPP 사업자에게 배분되는 수입입니다. 투자·유지·교체비를 빼기 전 금액입니다.",
  "20년 NPV":
    "20년간의 수입과 비용을 현재 가치로 환산한 순현재가치입니다. 양수면 보고서의 가정에서 투자비를 포함한 비용보다 수입의 현재 가치가 큽니다.",
  "운영 시뮬레이션":
    "‘준비됨’은 검증된 시간별 저장 결과가 있어 계획·현황·정산까지 살펴볼 수 있다는 뜻입니다. 실제 설비의 통신 연결 상태는 아닙니다.",
  "종료 SOC": `해당 시간이 끝날 때의 배터리 잔량입니다. ${socHelp}`,
  "계획 SOC": `계획대로 운영했을 때 예상한 배터리 잔량입니다. ${socHelp}`,
  "실행 SOC": `저장된 실행 결과의 배터리 잔량입니다. ${socHelp}`,
  "ESS 충전 / 방전":
    "충전은 전기를 배터리에 저장하는 것, 방전은 저장한 전기를 꺼내 쓰는 것입니다. 여기서는 하루 동안의 전력량을 합산합니다.",
  "ESS 충전": "이 시간에 배터리에 저장하도록 계획한 전력량입니다.",
  "ESS 방전": "이 시간에 배터리에서 꺼내 공급하도록 계획한 전력량입니다.",
  공급계획: supplyHelp,
  "선택 시간 공급계획": supplyHelp,
  "하루 공급계획":
    "하루 동안 전력망으로 공급하기로 계획한 전력량의 합계입니다. MWh는 전기의 양을 나타냅니다.",
  "실행 공급":
    "선택한 시간에 공급한 것으로 저장된 전력량입니다. 공급계획과 비교해 부족한 양을 확인할 수 있습니다.",
  "계획 기준 수입":
    "저장된 운영계획을 기준으로 계산한 수입입니다. ‘미계산’은 비교할 값이 저장되어 있지 않다는 뜻이며, 0원을 뜻하지 않습니다.",
  "ESS 귀속 에너지 수입":
    "ESS 운영과 관련해 VPP 사업자에게 배분된 에너지 거래 수입입니다. 편차 정산이 포함되어 있습니다.",
  "태양광 용량정산금 · 50%":
    "태양광의 공급 가능 용량에 대한 정산금 중 이 시나리오에서 VPP 사업자에게 배분한 몫입니다.",
  "임밸런스 페널티":
    "공급 약속과 실행 결과의 차이에 따라 이 시나리오에서 부과한 금액입니다. 음수는 수입에서 차감되는 금액입니다.",
  "판매 정산 · 편차 포함":
    "계획한 판매량의 거래 금액에 실제 공급량과의 차이에 대한 정산을 반영한 수입입니다.",
  "REC 수입":
    "재생에너지 공급인증서와 관련된 수입입니다. 이 화면은 저장된 실험의 적용 가정을 따릅니다.",
  "열화비 차감":
    "배터리를 충전·방전하면서 수명이 줄어드는 영향을 비용으로 환산한 값입니다.",
  "운영 순수익":
    "저장된 실험에서 전력 판매와 REC 수입을 더하고 배터리 열화비를 뺀 금액입니다. 설비 투자·금융비용은 제외됩니다.",
};

export function Help({ label }: { label: string }) {
  const description = descriptions[label];
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const tooltip = useRef<HTMLSpanElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ left: 0, top: 0 });
  const cancelClose = () => clearTimeout(timer.current);
  const show = () => {
    cancelClose();
    setOpen(true);
  };
  const closeSoon = () => {
    cancelClose();
    timer.current = setTimeout(() => setOpen(false), 160);
  };

  useLayoutEffect(() => {
    if (!open || !button.current || !tooltip.current) return;
    const anchor = button.current.getBoundingClientRect();
    const box = tooltip.current.getBoundingClientRect();
    setPosition({
      left: Math.max(
        12,
        Math.min(
          anchor.left + anchor.width / 2 - box.width / 2,
          window.innerWidth - box.width - 12,
        ),
      ),
      top:
        anchor.bottom + box.height + 8 < window.innerHeight - 12
          ? anchor.bottom + 8
          : Math.max(12, anchor.top - box.height - 8),
    });
    const close = () => setOpen(false);
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    const outside = (event: PointerEvent) => {
      if (
        !button.current?.contains(event.target as Node) &&
        !tooltip.current?.contains(event.target as Node)
      )
        close();
    };
    window.addEventListener("resize", close);
    document.addEventListener("scroll", close, true);
    document.addEventListener("keydown", escape);
    document.addEventListener("pointerdown", outside);
    return () => {
      cancelClose();
      window.removeEventListener("resize", close);
      document.removeEventListener("scroll", close, true);
      document.removeEventListener("keydown", escape);
      document.removeEventListener("pointerdown", outside);
    };
  }, [open]);

  if (!description) return null;
  return (
    <>
      <button
        ref={button}
        type="button"
        className="help-trigger"
        aria-label={`${label} 도움말`}
        aria-describedby={open ? id : undefined}
        onMouseEnter={show}
        onMouseLeave={closeSoon}
        onFocus={show}
        onBlur={() => setOpen(false)}
        onClick={show}
      >
        <span aria-hidden="true">?</span>
      </button>
      {open &&
        createPortal(
          <span
            ref={tooltip}
            id={id}
            role="tooltip"
            className="help-tooltip"
            style={position}
            onMouseEnter={cancelClose}
            onMouseLeave={closeSoon}
          >
            {description}
          </span>,
          document.body,
        )}
    </>
  );
}

export function Term({ label }: { label: string }) {
  return descriptions[label] ? (
    <span className="help-term">
      {label}
      <Help label={label} />
    </span>
  ) : (
    <>{label}</>
  );
}
