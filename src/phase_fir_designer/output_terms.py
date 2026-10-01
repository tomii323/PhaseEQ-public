"""Shared notice for generated ZIPs, separate from the application's license."""

from zipfile import ZipFile


OUTPUT_TERMS_FILENAME = "生成物の利用条件.txt"
OUTPUT_TERMS_TEXT = """PhaseEQ 生成物の利用条件（要約）
条件の制定日：2026年10月1日

PhaseEQを個人利用として使用して生成したFIR、WAV、BIN、TXT、CSV、FRD、
測定結果、設定データ等の成果物は、商用・業務用機器への組み込み、利用、
改変、販売、配布を含め、生成後は自由に利用できます。

ただし、商用・業務目的でPhaseEQ本体を使用して、新たに生成・調整することは
禁止されています。個人が操作することだけを理由に、業務目的の使用を
個人利用と扱うことはできません。

第三者のデータ・素材が含まれる場合は、それぞれの利用条件に従ってください。
生成物は無保証であり、法令が許す範囲で権利者は責任を負いません。

正式条件はPhaseEQのLICENSE.md（特に第6条・第7条・第9条）を参照してください。
https://github.com/tomii323/PhaseEQ-public/blob/main/LICENSE.md

この説明ファイルを生成物の再配布時にも添付する義務はありません。
本説明は生成物の利用に関する要約であり、PhaseEQ本体の商用利用や再配布を
許諾するものではありません。
"""
OUTPUT_TERMS_BYTES = OUTPUT_TERMS_TEXT.encode("utf-8-sig")


def write_output_terms(archive: ZipFile) -> None:
    """Attach the same UTF-8 notice without modifying generated payloads."""
    archive.writestr(OUTPUT_TERMS_FILENAME, OUTPUT_TERMS_BYTES)
