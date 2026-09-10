"""
Generates one lm-eval YAML task file per language, plus a group YAML
that lets you run all of them with a single task name: flores_plus.

Usage:
    cd lm_eval/tasks/flores_plus
    python create_configs.py

Then evaluate:
    lm_eval --model hf \
        --model_args pretrained=your-model \
        --tasks flores_plus \
        --include_path ./flores_plus \
        --device cuda \
        --output_path ./results
"""

import io
import os
from pathlib import Path

import pandas as pd


LANGS_MD = """
| Code  | Script | Glottocode | Name                                |                           Notes                            |
| ace | Arab | achi1257 | Acehnese (Jawi script)              |                                                            |
| ace | Latn | achi1257 | Acehnese (Latin script)             |                                                            |
| acm | Arab | meso1252 | Mesopotamian Arabic                 |                                                            |
| acq | Arab | taiz1242 | Taʽizzi-Adeni Arabic                |                                                            |
| aeb | Arab | tuni1259 | Tunisian Arabic                     |                                                            |
| afr | Latn | afri1274 | Afrikaans                           |                                                            |
| als | Latn | tosk1239 | Albanian (Tosk)                     |                                                            |
| amh | Ethi | amha1245 | Amharic                             |                                                            |
| apc | Arab | nort3139 | Levantine Arabic (North)            |                                                            |
| apc | Arab | sout3123 | Levantine Arabic (South)            |                                                            |
| apd | Arab | khar1296 | [Sudanese Arabic](dataset_cards/apd_Arab.md)     |                                               |
| arb | Arab | stan1318 | Modern Standard Arabic              |                                                            |
| arb | Latn | stan1318 | Modern Standard Arabic (Romanized)  |                                                            |
| arg | Latn | arag1245 | [Aragonese](dataset_cards/arg_Latn.md)          |                                                |
| ars | Arab | najd1235 | Najdi Arabic                        |                                                            |
| ary | Arab | moro1292 | Moroccan Arabic                     |                                                            |
| arz | Arab | egyp1253 | Egyptian Arabic                     |                                                            |
| asm | Beng | assa1263 | Assamese                            |                                                            |
| ast | Latn | astu1245 | [Asturian](dataset_cards/ast_Latn.md)           |                                                |
| awa | Deva | awad1243 | Awadhi                              |                                                            |
| ayr | Latn | cent2142 | Central Aymara                      |                                                            |
| azb | Arab | sout2697 | South Azerbaijani                   |                                                            |
| azj | Latn | nort2697 | North Azerbaijani                   |                                                            |
| bak | Cyrl | bash1264 | Bashkir                             |                                                            |
| bam | Latn | bamb1269 | Bambara                             |                                                            |
| ban | Latn | bali1278 | Balinese                            |                                                            |
| bel | Cyrl | bela1254 | Belarusian                          |                                                            |
| bem | Latn | bemb1257 | Bemba                               |                                                            |
| ben | Beng | beng1280 | Bengali                             |                                                            |
| bho | Deva | bhoj1244 | Bhojpuri                            |                                                            |
| bjn | Arab | banj1239 | Banjar (Jawi script)                |                                                            |
| bjn | Latn | banj1239 | Banjar (Latin script)               |                                                            |
| bod | Tibt | utsa1239 | Lhasa Tibetan                       |                                                            |
| bos | Latn | bosn1245 | Bosnian                             |                                                            |
| brx | Deva | bodo1269 | Bodo                                |                         dev only                         |
| bug | Latn | bugi1244 | Buginese                            |                                                            |
| bul | Cyrl | bulg1262 | Bulgarian                           |                                                            |
| cat | Latn | stan1289 | Catalan                             |                                                            |
| cat | Latn | vale1252 | [Valencian](dataset_cards/cat_Latn_vale1252.md) |          devtest only                        |
| ceb | Latn | cebu1242 | Cebuano                             |                                                            |
| ces | Latn | czec1258 | Czech                               |                                                            |
| chv | Cyrl | chuv1255 | [Chuvash](dataset_cards/chv_Cyrl.md)|                                                            |
| cjk | Latn | chok1245 | Chokwe                              |                                                            |
| ckb | Arab | cent1972 | Central Kurdish                     |                                                            |
| cmn | Hans | beij1234 | [Mandarin Chinese (Standard Beijing)](dataset_cards/cmn_Hans.md) |                               |
| cmn | Hant | taib1240 | [Mandarin Chinese (Taiwanese)](dataset_cards/cmn_Hant.md)       |                                |
| crh | Latn | crim1257 | Crimean Tatar                       |                                                            |
| cym | Latn | wels1247 | Welsh                               |                                                            |
| dan | Latn | dani1285 | Danish                              |                                                            |
| dar | Cyrl | darg1241 | [Dargwa](dataset_cards/dar_Cyrl.md) |                         dev only                         |
| deu | Latn | stan1295 | German                              |                                                            |
| dgo | Deva | dogr1250 | Dogri                               |                         dev only                         |
| dik | Latn | sout2832 | Southwestern Dinka                  |                                                            |
| dyu | Latn | dyul1238 | Dyula                               |                                                            |
| dzo | Tibt | dzon1239 | Dzongkha                            |                                                            |
| ekk | Latn | esto1258 | Estonian                            |                                                            |
| ell | Grek | mode1248 | Greek                               |                                                            |
| eng | Latn | stan1293 | English                             |                                                            |
| epo | Latn | espe1235 | Esperanto                           |                                                            |
| eus | Latn | basq1248 | Basque                              |                                                            |
| ewe | Latn | ewee1241 | Ewe                                 |                                                            |
| fao | Latn | faro1244 | Faroese                             |                                                            |
| fij | Latn | fiji1243 | Fijian                              |                                                            |
| fil | Latn | fili1244 | [Filipino](dataset_cards/fil_Latn.md)           |                                                |
| fin | Latn | finn1318 | Finnish                             |                                                            |
| fon | Latn | fonn1241 | Fon                                 |                                                            |
| fra | Latn | stan1290 | French                              |                                                            |
| fur | Latn | east2271 | Friulian                            |                                                            |
| fuv | Latn | nige1253 | Nigerian Fulfulde                   |                                                            |
| gaz | Latn | west2721 | West Central Oromo                  |                                                            |
| gla | Latn | scot1245 | Scottish Gaelic                     |                                                            |
| gle | Latn | iris1253 | Irish                               |                                                            |
| glg | Latn | gali1258 | Galician                            |                                                            |
| gom | Deva | goan1235 | Goan Konkani                        |                         dev only                         |
| gug | Latn | para1311 | Paraguayan Guaraní                  |                                                            |
| guj | Gujr | guja1252 | Gujarati                            |                                                            |
| hat | Latn | hait1244 | Haitian Creole                      |                                                            |
| hau | Latn | haus1257 | Hausa                               |                                                            |
| heb | Hebr | hebr1245 | Hebrew                              |                                                            |
| hin | Deva | hind1269 | Hindi                               |                                                            |
| hne | Deva | chha1249 | Chhattisgarhi                       |                                                            |
| hrv | Latn | croa1245 | Croatian                            |                                                            |
| hun | Latn | hung1274 | Hungarian                           |                                                            |
| hye | Armn | nucl1235 | Armenian                            |                                                            |
| ibo | Latn | nucl1417 | Igbo                                |                                                            |
| ilo | Latn | ilok1237 | Ilocano                             |                                                            |
| ind | Latn | indo1316 | Indonesian                          |                                                            |
| isl | Latn | icel1247 | Icelandic                           |                                                            |
| ita | Latn | ital1282 | Italian                             |                                                            |
| jav | Latn | java1254 | Javanese                            |                                                            |
| jpn | Jpan | nucl1643 | Japanese                            |                                                            |
| kaa | Latn | kara1467 | [Karakalpak](dataset_cards/kaa_Latn.md)         |           devtest only                       |
| kab | Latn | kaby1243 | Kabyle                              |                                                            |
| kac | Latn | kach1280 | Jingpho                             |                                                            |
| kam | Latn | kamb1297 | Kamba                               |                                                            |
| kan | Knda | nucl1305 | Kannada                             |                                                            |
| kas | Arab | kash1277 | Kashmiri (Arabic script)            |                                                            |
| kas | Deva | kash1277 | Kashmiri (Devanagari script)        |                                                            |
| kat | Geor | nucl1302 | Georgian                            |                                                            |
| kaz | Cyrl | kaza1248 | Kazakh                              |                                                            |
| kbp | Latn | kabi1261 | Kabiyè                              |                                                            |
| kea | Latn | kabu1256 | Kabuverdianu                        |                                                            |
| khk | Cyrl | halh1238 | Halh Mongolian (Cyrillic script)    |                                                            |
| khk | Mong | halh1238 | [Halh Mongolian (traditional Mongolian script)](dataset_cards/khk_Mong.md)| devtest only     |
| khm | Khmr | cent1989 | Khmer (Central)                     |                                                            |
| kik | Latn | kiku1240 | Kikuyu                              |                                                            |
| kin | Latn | kiny1244 | Kinyarwanda                         |                                                            |
| kir | Cyrl | kirg1245 | Kyrgyz                              |                                                            |
| kjh | Cyrl | khak1248 | [Khakas](dataset_cards/kjh_Cyrl.md) |                                                            |
| kmb | Latn | kimb1241 | Kimbundu                            |                                                            |
| kmr | Latn | nort2641 | Northern Kurdish                    |                                                            |
| knc | Arab | cent2050 | Central Kanuri (Arabic script)      |                                                            |
| knc | Latn | cent2050 | Central Kanuri (Latin script)       |                                                            |
| kor | Hang | kore1280 | Korean                              |                                                            |
| ktu | Latn | kitu1246 | Kituba (DRC)                        |                                                            |
| lao | Laoo | laoo1244 | Lao                                 |                                                            |
| lij | Latn | geno1240 | Ligurian (Genoese)                  |                                                            |
| lim | Latn | limb1263 | Limburgish                          |                                                            |
| lin | Latn | ling1263 | Lingala                             |                                                            |
| lit | Latn | lith1251 | Lithuanian                          |                                                            |
| lld | Latn | ladi1250 | [Ladin (Val Badia)](dataset_cards/lld_Latn.md)  |                                                |
| lld | Latn | gard1241 | [Ladin (Gherdëina)](dataset_cards/lld_Latn_gard1241.md)  |                                       |
| lmo | Latn | lomb1257 | Lombard                             | [[1]](https://github.com/openlanguagedata/flores/issues/5) |
| ltg | Latn | east2282 | Latgalian                           |                                                            |
| ltz | Latn | luxe1241 | Luxembourgish                       |                                                            |
| lua | Latn | luba1249 | Luba-Kasai                          |                                                            |
| lug | Latn | gand1255 | Ganda                               |                                                            |
| luo | Latn | luok1236 | Luo                                 |                                                            |
| lus | Latn | lush1249 | Mizo                                |                                                            |
| lvs | Latn | stan1325 | Standard Latvian                    |                                                            |
| mag | Deva | maga1260 | Magahi                              |                                                            |
| mai | Deva | mait1250 | Maithili                            |                                                            |
| mal | Mlym | mala1464 | Malayalam                           |                                                            |
| mar | Deva | mara1378 | Marathi                             |                                                            |
| mfe | Latn | mori1278 | [Mauritian Creole](dataset_cards/mfe_Latn.md)   |                                                |
| mhr | Cyrl | gras1239 | [Meadow Mari](dataset_cards/mhr_Cyrl.md)        |                                                |
| min | Arab | mina1268 | Minangkabau (Jawi script)           |                                                            |
| min | Latn | mina1268 | Minangkabau (Latin script)          |                                                            |
| mkd | Cyrl | mace1250 | Macedonian                          |                                                            |
| mlt | Latn | malt1254 | Maltese                             |                                                            |
| mni | Beng | mani1292 | Meitei (Manipuri, Bengali script)   |                                                            |
| mni | Mtei | mani1292 | Meitei (Manipuri, Meitei script)    |                         dev only                         |
| mos | Latn | moss1236 | Mossi                               |                                                            |
| mri | Latn | maor1246 | Maori                               |                                                            |
| mya | Mymr | nucl1310 | Burmese                             |                                                            |
| myv | Cyrl | erzy1239 | [Erzya](dataset_cards/myv_Cyrl.md)  |                                                            |
| nld | Latn | dutc1256 | Dutch                               |                                                            |
| nno | Latn | norw1262 | Norwegian Nynorsk                   |                                                            |
| nob | Latn | norw1259 | [Norwegian Bokmål, moderate variety](dataset_cards/nob_Latn.md)  |                               |
| nob | Latn | norw1259 (with variant=radical) | [Norwegian Bokmål, radical variety](dataset_cards/nob_Latn_radical.md)  | |
| npi | Deva | nepa1254 | Nepali                              |                                                            |
| nqo | Nkoo | nkoa1234 | Nko                                 |                                                            |
| nso | Latn | pedi1238 | Northern Sotho                      |                                                            |
| nus | Latn | nuer1246 | Nuer                                |                                                            |
| nya | Latn | nyan1308 | Nyanja                              |                                                            |
| oci | Latn | occi1239 | Occitan                             |                                                            |
| oci | Latn | aran1260 | [Aranese](dataset_cards/oci_Latn_aran1260.md)   |                                                |
| ory | Orya | oriy1255 | Odia                                |                                                            |
| pag | Latn | pang1290 | Pangasinan                          |                                                            |
| pan | Guru | panj1256 | Eastern Panjabi                     |                                                            |
| pap | Latn | papi1253 | Papiamento                          |                                                            |
| pbt | Arab | sout2649 | Southern Pashto                     |                                                            |
| pes | Arab | west2369 | Western Persian                     |                                                            |
| plt | Latn | plat1254 | Plateau Malagasy                    |                                                            |
| pol | Latn | poli1260 | Polish                              |                                                            |
| por | Latn | braz1246 | Portuguese (Brazilian)              |                                                            |
| prs | Arab | dari1249 | Dari                                |                                                            |
| quy | Latn | ayac1239 | Ayacucho Quechua                    |                                                            |
| ron | Latn | roma1327 | Romanian                            |                                                            |
| run | Latn | rund1242 | Rundi                               |                                                            |
| rus | Cyrl | russ1263 | Russian                             |                                                            |
| sag | Latn | sang1328 | Sango                               |                                                            |
| san | Deva | sans1269 | Sanskrit                            |                                                            |
| sat | Olck | sant1410 | Santali                             |                                                            |
| scn | Latn | sici1248 | Sicilian                            |                                                            |
| shn | Mymr | shan1277 | Shan                                |                                                            |
| sin | Sinh | sinh1246 | Sinhala                             |                                                            |
| slk | Latn | slov1269 | Slovak                              |                                                            |
| slv | Latn | slov1268 | Slovenian                           |                                                            |
| smo | Latn | samo1305 | Samoan                              |                                                            |
| sna | Latn | shon1251 | Shona                               |                                                            |
| snd | Arab | sind1272 | Sindhi (Arabic script)              |                                                            |
| snd | Deva | sind1272 | Sindhi (Devanagari script)          |                         dev only                         |
| som | Latn | soma1255 | Somali                              |                                                            |
| sot | Latn | sout2807 | Southern Sotho                      |                                                            |
| spa | Latn | amer1254 | Spanish (Latin American)            |                                                            |
| srd | Latn | sard1257 | Sardinian                           | [[1]](https://github.com/openlanguagedata/flores/issues/6) |
| srp | Cyrl | serb1264 | Serbian                             |                                                            |
| ssw | Latn | swat1243 | Swati                               |                                                            |
| sun | Latn | sund1252 | Sundanese                           |                                                            |
| swe | Latn | swed1254 | Swedish                             |                                                            |
| swh | Latn | swah1253 | Swahili                             |                                                            |
| szl | Latn | sile1253 | Silesian                            |                                                            |
| tam | Taml | tami1289 | Tamil                               |                                                            |
| taq | Latn | tama1365 | Tamasheq (Latin script)             |                                                            |
| taq | Tfng | tama1365 | Tamasheq (Tifinagh script)          |                                                            |
| tat | Cyrl | tata1255 | Tatar                               |                                                            |
| tel | Telu | telu1262 | Telugu                              |                                                            |
| tgk | Cyrl | taji1245 | Tajik                               |                                                            |
| tha | Thai | thai1261 | Thai                                |                                                            |
| tir | Ethi | tigr1271 | Tigrinya                            |                                                            |
| tpi | Latn | tokp1240 | Tok Pisin                           |                                                            |
| tsn | Latn | tswa1253 | Tswana                              |                                                            |
| tso | Latn | tson1249 | Tsonga                              |                                                            |
| tuk | Latn | turk1304 | Turkmen                             |                                                            |
| tum | Latn | tumb1250 | Tumbuka                             |                                                            |
| tur | Latn | nucl1301 | Turkish                             |                                                            |
| twi | Latn | akua1239 | Akuapem Twi                         |                                                            |
| twi | Latn | asan1239 | Asante Twi                          |                                                            |
| tyv | Cyrl | tuvi1240 | [Tuvan](dataset_cards/tyv_Cyrl.md)  |                                                            |
| udm | Cyrl | udmu1245 | [Udmurt](dataset_cards/udm_Cyrl.md) |                           dev only                       |
| uig | Arab | uigh1240 | Uyghur                              |                                                            |
| ukr | Cyrl | ukra1253 | Ukrainian                           |                                                            |
| umb | Latn | umbu1257 | Umbundu                             |                                                            |
| urd | Arab | urdu1245 | Urdu                                |                                                            |
| uzn | Latn | nort2690 | Northern Uzbek                      |                                                            |
| uzs | Arab | sout2699 | [Southern Uzbek](dataset_cards/uzs_Arab.md) |                   dev only                       |
| vec | Latn | vene1259 | Venetian                            |                                                            |
| vie | Latn | viet1252 | Vietnamese                          |                                                            |
| vmw | Latn | cent2033 | [Emakhuwa (Central)](dataset_cards/vmw_Latn.md) |                                                |
| war | Latn | wara1300 | Waray                               |                                                            |
| wol | Latn | nucl1347 | Wolof                               |                                                            |
| wuu | Hans | suhu1238 | [Wu Chinese](dataset_cards/wuu_Hans.md)         |               dev only                       |
| xho | Latn | xhos1239 | Xhosa                               |                                                            |
| ydd | Hebr | east2295 | Eastern Yiddish                     |                                                            |
| yor | Latn | yoru1245 | Yoruba                              |                                                            |
| yue | Hant | xian1255 | [Yue Chinese (Hong Kong Cantonese)](dataset_cards/yue_Hant.md)   |                               |
| zgh | Tfng | stan1324 | [Standard Moroccan Tamazight](dataset_cards/zgh_Tfng.md)         |                               |
| zsm | Latn | stan1306 | Standard Malay                      |                                                            |
| zul | Latn | zulu1248 | Zulu                                |                                                            |
"""


# lm-eval computes bits-per-byte from loglikelihood_rolling as:
#   bpb = -loglikelihood / (num_bytes * log(2))
# word_perplexity and byte_perplexity are also reported automatically.
TASK_TEMPLATE = """\
task: flores_plus_{lang}
dataset_path: {repo_id}
dataset_name: {lang}
test_split: devtest
validation_split: dev
# doc_to_text: "{{{{sentence}}}}"
# doc_to_target: ""
doc_to_text: ""
doc_to_target: "{{{{sentence}}}}"
description: "FLORES-200 {lang} — rolling log-likelihood for BPB / perplexity."
output_type: loglikelihood_rolling
metadata:
  version: 1.0
"""

GROUP_TEMPLATE = """\
group: flores_plus
task:
{task_list}
metadata:
  version: 1.0
"""


def main(repo_id: str, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)

    task_names = []
    df = pd.read_csv(
        io.StringIO(LANGS_MD),
        sep="|",
        header=0,
        skipinitialspace=True,
        skip_blank_lines=True,
    ).dropna(axis=1, how="all")
    df.columns = df.columns.str.strip()
    df = df.apply(
        lambda col: col.str.strip() if col.dtype in ["object", "str"] else col
    )
    df["iso_code"] = df["Code"] + "_" + df["Script"]
    # import code; code.interact(local=dict(globals(), **locals()))

    for lang in df["iso_code"]:
        task_name = f"flores200_{lang}"
        task_names.append(task_name)
        yaml_path = out_dir / f"{task_name}.yaml"
        with open(yaml_path, "w") as f:
            f.write(TASK_TEMPLATE.format(lang=lang, repo_id=repo_id))
        print(f"  wrote {yaml_path}")

    # Group YAML so --tasks flores200 runs all languages at once
    task_list_str = "\n".join(f"  - {t}" for t in task_names)
    group_path = out_dir / "_flores200_group.yaml"
    with open(group_path, "w") as f:
        f.write(GROUP_TEMPLATE.format(task_list=task_list_str))
    print(f"  wrote {group_path}")

    print("\nDone. Run with:")
    print("  lm_eval --model hf \\")
    print("      --model_args pretrained=<your-model> \\")
    print("      --tasks flores200 \\")
    print(f"      --include_path {out_dir} \\")
    print("      --device cuda \\")
    print("      --output_path ./results")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    repo = "openlanguagedata/flores_plus"
    main(repo_id=repo, out_dir=out_dir)
