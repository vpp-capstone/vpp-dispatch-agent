import { ArrowRight, ArrowUpRight } from "lucide-react";

export function Welcome({
  hasSelection,
  canResume,
  onStart,
  onResume,
}: {
  hasSelection: boolean;
  canResume: boolean;
  onStart: () => void;
  onResume: () => void;
}) {
  return (
    <div className="welcome">
      <section className="welcome-intro">
        <div className="eyebrow">제주 VPP 중개사업자를 위한 의사결정 공간</div>
        <h1>
          처음 오셨나요?
          <br />
          설비 검토부터
          <br />
          <em>운영의 첫날까지.</em>
        </h1>
        <p className="welcome-description">
          우리 사업에 맞는 설비와 수익성을 먼저 살펴보세요.
          <br />
          선택한 조건을 운영계획으로 연결하고, 실제 결과를 함께 확인합니다.
        </p>
        <button className="button primary welcome-start" onClick={onStart}>
          설비·사업성 검토 시작 <ArrowRight size={18} />
        </button>
        <p className="caption">
          저장된 분석 결과로 고객의 의사결정 과정을 체험합니다.
        </p>
      </section>
      <aside className="welcome-journey" aria-label="처음 이용하는 고객의 흐름">
        <div className="eyebrow">검토한 조건이 운영의 기준이 됩니다</div>
        <ol>
          <li>
            <span>01</span>
            <div>
              <h2>설비·사업성 검토</h2>
              <p>선로 조건과 설비 후보의 수입·NPV를 비교합니다.</p>
            </div>
          </li>
          <li>
            <span>02</span>
            <div>
              <h2>선택안 확정</h2>
              <p>검토한 후보를 시연의 운영 대상으로 선택합니다.</p>
            </div>
          </li>
          <li>
            <span>03</span>
            <div>
              <h2>운영계획 → 현황 → 실적·정산</h2>
              <p>같은 설비의 계획, 실행, 수입과 차이를 추적합니다.</p>
            </div>
          </li>
        </ol>
        <div className="welcome-returning">
          <h2>이미 운영 중이신가요?</h2>
          <p>
            {hasSelection
              ? "이 브라우저에 저장된 운영 대상으로 바로 이어갑니다."
              : "기존 고객의 일상 업무를 예시 설비로 확인하세요."}
          </p>
          <button
            className="text-button"
            disabled={!canResume}
            onClick={onResume}
          >
            {hasSelection ? "기존 운영계획 열기" : "기존 고객으로 시연하기"}{" "}
            <ArrowUpRight size={15} />
          </button>
        </div>
      </aside>
    </div>
  );
}
