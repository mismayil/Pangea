## PANGEAINS

The PANGEAINS dataset is designed to address the scarcity of multilingual multimodal instructions, incorporating tasks and data from diverse linguistic and cultural contexts. It includes:

- **General instructions** (e.g., VQA)
- **Document and chart question answering**
- **Captioning**
- **Cultural-relevant tasks**
- **Domain-specific tasks**
- **text-only instructions**

Here is a summary of the dataset structure and distribution:

![PANGEAINS Overview](data/data_distribution.png)

The image above provides an overview of the dataset, which includes:

- 3.4 million general multimodal samples
- 325K document and chart tasks
- 100K captioning tasks
- 1.5 million culturally relevant samples
- 60K domain-specific tasks
- 820K text-only tasks


### PANGEAINS Sample Data

This repository contains a sample of the PANGEAINS dataset, a multilingual multimodal instruction dataset that spans 39 languages and a wide variety of tasks. Due to size limitations, we are only providing a subset of the full dataset here. The complete dataset, which includes 6 million multimodal samples, will be released in full at a later date.

## Data Format

The sample data is structured in JSON format, where each entry contains an image reference and a corresponding instruction-response pair. Below is an example from the dataset:

```json
{
   "id": 2433684022797.0,
   "image": "images/cultural/2433684022797.0.jpg",
   "conversations": [
      {
         "from": "human",
         "value": "<image>\n사진 속의 단풍잎들로 어떤 예술 작품을 만들 수 있을까? 예술 작품의 구체적인 아이디어와 그 이유를 상세히 설명해보세요."
      },
      {
         "from": "gpt",
         "value": "사진 속의 붉고 화려한 단풍잎들을 이용하여 '모자이크 예술 작품'을 만들어 보는 것은 어떨까요? 먼저, 단풍잎들을 작은 조각으로 잘라 내어 다양한 색상과 질감을 갖도록 준비합니다. 그런 다음, 이 조각들을 캔버스 위에 붙여 모자이크 패턴을 형성합니다. 예를 들어, 단풍잎 조각들을 이용하여 나뭇잎 모양의 패턴이나 추상적인 예술 작품을 만들 수 있습니다. 이러한 작업을 통해 자연의 아름다움과 계절의 순환을 상징하는 단풍을 예술 작품으로 재해석할 수 있습니다. 모자이크 작업은 단순한 예술 작품을 넘어, 환경 친화적이고 지속 가능한 예술 작업을 촉진하는 데에도 기여할 수 있습니다."
      },
      {
         "from": "human",
         "value": "사진 속의 단풍잎들을 이용하여 어떤 행사를 기획한다면, 어떤 행사를 기획하고 어떻게 준비해야 할지 구체적으로 설명해보세요."
      },
      {
         "from": "gpt",
         "value": "사진 속의 단풍잎들을 이용하여 '가을 단풍 축제'를 기획해 보는 것은 어떨까요? 먼저, 축제 장소를 한국의 아름다운 자연 경관이 있는 공원이나 산에서 열도록 준비합니다. 단풍잎들을 이용하여 장식을 만들고, 가을 분위기를 연출하는 데 사용할 수 있습니다. 또한, 단풍놀이, 산책, 사진 촬영 대회 등 다양한 이벤트를 포함하여 참가자들이 자연을 즐기고 가을의 아름다움을 느낄 수 있도록 합니다. 음식 및 음료 부스도 설치하여 현지 특산물이나 가을 음식을 제공합니다. 마케팅을 위해 소셜 미디어를 활용하여 축제를 홍보하고, 지역 공동체와의 협력을 통해 행사의 성공을 보장합니다. 이러한 준비를 통해 단풍이 가져다주는 자연의 아름다움과 문화적 의미를 참가자들과 함께 나눌 수 있는 기회를 제공합니다."
      }
   ],
   "language": "ko"
}
```
![ex](data/images/cultural/2433684022797.0.jpg)

In this example, the image is linked from the `images` directory, and the task involves identifying the time of day based on the provided image.

## Directory Structure

The sample dataset is organized as follows:

```
data/
  ├── sample_data.json        # The sample data in JSON format
  └── images/                 # The directory containing corresponding images
```

## Full Dataset Release

This sample represents a small fraction of the full PANGEAINS dataset, which consists of 6 million instructions. The complete dataset will be made available in the future, and will include expanded tasks, more diverse cultural contexts, and a wider range of languages.
