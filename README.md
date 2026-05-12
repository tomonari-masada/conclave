# CONCLAVE
* CONCLAVE（CONcept Crystalization with LAtent VEctors）
  * 元はConCryST (Consept Crystalization with Soft Tokens) という名前にしていたが、soft token以外の可能性も考えているので、改名した。コードはsoft tokenを利用した手法を実装したもの。

* 関連しそうな研究（要調査）
  * Lester et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." EMNLP 2021.
→ Soft prompt tuning の基礎。モデル固定・入力側ベクトルのみ学習という設計の出発点。
  * Li & Liang (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL 2021.
→ 全層へのprefixによる生成制御。CONCLAVEの入力介入アプローチの先行研究。
  * Turner et al. (2023). "Steering Language Models With Activation Engineering." arXiv:2308.10248.
→ Activation Addition (ActAdd): 活性化空間へのステアリングベクトル加算。CONCLAVEが視野に入れるsteering vector手法の代表的先行研究。
  * Postmus & Abreu (2024). "Steering Large Language Models using Conceptors." arXiv:2410.16314.
→ 概念をconceptorとして表現し活性化空間を操作。概念の構造化表現という観点でCONCLAVEと問題意識が近い。
  * Qian et al. (2022). "Controllable Natural Language Generation with Contrastive Prefixes." ACL 2022 Findings.
→ 対照学習とprefixを組み合わせた制御可能なテキスト生成。CONCLAVEの対照学習設計と関連。
