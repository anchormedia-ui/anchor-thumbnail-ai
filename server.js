import express from 'express';
import cors from 'cors';
import multer from 'multer';
import dotenv from 'dotenv';
import sharp from 'sharp';
import { GoogleGenAI, Modality } from '@google/genai';

dotenv.config();

const app = express();

const PORT = process.env.PORT || 3000;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || '*';

if (!GEMINI_API_KEY) {
  console.warn('WARNING: GEMINI_API_KEY is missing from .env');
}

const ai = new GoogleGenAI({
  apiKey: GEMINI_API_KEY
});

app.use(cors({
  origin: ALLOWED_ORIGIN === '*' ? true : ALLOWED_ORIGIN
}));

app.use(express.json({
  limit: '2mb'
}));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 15 * 1024 * 1024,
    files: 25
  }
});

const MODEL = 'gemini-2.5-flash-image';

const OUTPUTS = {
  main: {
    width: 1280,
    height: 720,
    aspectLabel: '16:9'
  },
  noText: {
    width: 1280,
    height: 720,
    aspectLabel: '16:9'
  },
  mobile: {
    width: 1080,
    height: 1350,
    aspectLabel: '4:5'
  }
};

app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    service: 'Anchor Thumbnail AI',
    model: MODEL
  });
});

app.post(
  '/api/generate-thumbnails',
  upload.fields([
    { name: 'speakerPhoto', maxCount: 1 },
    { name: 'supportPhotos', maxCount: 8 },
    { name: 'inspirationPhotos', maxCount: 16 }
  ]),
  async (req, res) => {
    try {
      const form = normalizeForm(req.body || {});
      const files = req.files || {};

      const speakerPhoto = files.speakerPhoto?.[0];
      const supportPhotos = files.supportPhotos || [];
      const inspirationPhotos = files.inspirationPhotos || [];

      if (!form.messageTitle) {
        return fail(res, 400, 'Message Title is required.');
      }

      if (!form.thumbnailText) {
        return fail(res, 400, 'Short Phrase / Thumbnail Text is required.');
      }

      if (!form.speakerName) {
        return fail(res, 400, 'Speaker Name is required.');
      }

      if (!speakerPhoto) {
        return fail(res, 400, 'Main Speaker Photo is required.');
      }

      const references = buildReferences({
        speakerPhoto,
        supportPhotos,
        inspirationPhotos
      });

      const mainPrompt = buildMainPrompt(form);
      const noTextPrompt = buildNoTextPrompt(form);
      const mobilePrompt = buildMobilePrompt(form);

      const [mainRaw, noTextRaw, mobileRaw] = await Promise.all([
        generateGeminiImage({
          prompt: mainPrompt,
          references
        }),
        generateGeminiImage({
          prompt: noTextPrompt,
          references
        }),
        generateGeminiImage({
          prompt: mobilePrompt,
          references
        })
      ]);

      const mainPng = await forceSize({
        imageBuffer: mainRaw.buffer,
        width: OUTPUTS.main.width,
        height: OUTPUTS.main.height
      });

      const noTextPng = await forceSize({
        imageBuffer: noTextRaw.buffer,
        width: OUTPUTS.noText.width,
        height: OUTPUTS.noText.height
      });

      const mobilePng = await forceSize({
        imageBuffer: mobileRaw.buffer,
        width: OUTPUTS.mobile.width,
        height: OUTPUTS.mobile.height
      });

      res.json({
        ok: true,
        thumbnails: {
          main: {
            width: OUTPUTS.main.width,
            height: OUTPUTS.main.height,
            url: bufferToDataUrl(mainPng, 'image/png')
          },
          noText: {
            width: OUTPUTS.noText.width,
            height: OUTPUTS.noText.height,
            url: bufferToDataUrl(noTextPng, 'image/png')
          },
          mobile: {
            width: OUTPUTS.mobile.width,
            height: OUTPUTS.mobile.height,
            url: bufferToDataUrl(mobilePng, 'image/png')
          }
        }
      });
    } catch (error) {
      console.error('Thumbnail generation failed:', error);
      return fail(res, 500, error.message || 'Thumbnail generation failed.');
    }
  }
);

function normalizeForm(body) {
  return {
    messageTitle: clean(body.messageTitle),
    thumbnailText: clean(body.thumbnailText),
    speakerName: clean(body.speakerName),
    speakerRole: clean(body.speakerRole),
    scriptureTopic: clean(body.scriptureTopic),
    stylePreset: clean(body.stylePreset) || 'bold-sermon',
    extraNotes: clean(body.extraNotes)
  };
}

function clean(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function buildReferences({ speakerPhoto, supportPhotos, inspirationPhotos }) {
  const refs = [];

  refs.push({
    type: 'main speaker photo',
    name: speakerPhoto.originalname,
    mimeType: speakerPhoto.mimetype,
    base64: speakerPhoto.buffer.toString('base64')
  });

  supportPhotos.forEach((file, index) => {
    refs.push({
      type: `support photo ${index + 1}`,
      name: file.originalname,
      mimeType: file.mimetype,
      base64: file.buffer.toString('base64')
    });
  });

  inspirationPhotos.forEach((file, index) => {
    refs.push({
      type: `style inspiration ${index + 1}`,
      name: file.originalname,
      mimeType: file.mimetype,
      base64: file.buffer.toString('base64')
    });
  });

  return refs;
}

function getStyleDescription(stylePreset) {
  const styles = {
    'bold-sermon': `
Bold sermon thumbnail style.
Use strong typography, dramatic contrast, layered poster composition, real preacher cutout, premium church media look, and subtle paper/grain texture.
`,

    'riot-youth': `
Riot Youth style.
Energetic, bold, young, gritty poster design, strong typography, graphic scribbles, texture, high contrast, and modern Pinterest-inspired church youth design.
`,

    'text-graphic': `
Text and graphic style.
The words and graphics should carry the design. Use symbols, abstract shapes, paper texture, strong type hierarchy, and avoid using fake people.
`,

    'modern-church': `
Modern church style.
Clean but bold, editorial layout, premium typography, strong visual hierarchy, real preacher photo if provided, and polished church design.
`,

    'clean-minimal': `
Clean minimal style.
Simple, premium, refined spacing, fewer graphic elements, clean type, soft texture, and strong focus.
`
  };

  return styles[stylePreset] || styles['bold-sermon'];
}

function buildBasePrompt(form) {
  return `
You are creating a finished professional sermon thumbnail for Anchor Church.

Use the uploaded MAIN SPEAKER PHOTO as the real speaker. Preserve the real person's likeness. Do not invent or replace the face.

Important people rules:
- Do not create fake random people.
- Do not add fake crowds or fake preachers.
- If a person is visible, it must be based on the uploaded real speaker photo or uploaded support photos.
- If extra people are needed but no real photo is provided, use silhouettes, shadows, outlines, or abstract graphic figures only.

Design style:
${getStyleDescription(form.stylePreset)}

User-provided inspiration images:
Use them for visual direction only. Do not copy them exactly. Pull from their style traits such as typography, texture, color blocking, layout energy, poster grain, graphic accents, and composition.

Anchor Church thumbnail preferences:
- Modern Pinterest-style church design.
- Real church media look.
- No generic AI church background.
- No cheesy clip art.
- No fake people.
- Strong bold typography when text is requested.
- Poster texture, grain, halftone, paper, scribbles, arrows, shapes, layered cutouts are allowed.
- Keep it clean enough to read fast.
- Make it look finished and publish-ready.

Message Title: ${form.messageTitle}
Main Thumbnail Text: ${form.thumbnailText}
Speaker: ${[form.speakerRole, form.speakerName].filter(Boolean).join(' ')}
Scripture / Topic: ${form.scriptureTopic || 'Not provided'}
Extra Notes: ${form.extraNotes || 'None'}
`.trim();
}

function buildMainPrompt(form) {
  return `
${buildBasePrompt(form)}

Create OUTPUT 1: MAIN YOUTUBE THUMBNAIL.

Canvas:
- 16:9 horizontal thumbnail.
- Final file will be resized to 1280x720.

Must include text:
- Use this as the main large readable thumbnail text: "${form.thumbnailText}"
- Keep the main text short, bold, and readable.
- You may include the speaker name in a smaller tasteful tag if it improves the design.
- Do not overload the design with too many words.

Composition:
- Use the real uploaded speaker photo prominently.
- Make the person feel integrated into the design, not pasted randomly.
- Use strong poster-like layout inspired by the reference images.
- Make it strong enough for YouTube.
- Leave safe margins so important words are not cut off.
`.trim();
}

function buildNoTextPrompt(form) {
  return `
${buildBasePrompt(form)}

Create OUTPUT 2: WEBSITE NO-TEXT THUMBNAIL.

Canvas:
- 16:9 horizontal thumbnail.
- Final file will be resized to 1280x720.

Text rules:
- Do not include any words, letters, captions, logos, fake text, typography, or readable marks.
- No title text.
- No speaker name.
- No scripture text.

Composition:
- Push the main subject and strongest visual elements to the RIGHT side.
- Keep the LEFT side cleaner and more open because website text will be placed there later.
- Use the real uploaded speaker photo if it works.
- If using the speaker, place the speaker mostly on the right half.
- Make the image still feel designed and premium without text.
`.trim();
}

function buildMobilePrompt(form) {
  return `
${buildBasePrompt(form)}

Create OUTPUT 3: MOBILE VERTICAL THUMBNAIL.

Canvas:
- 4:5 vertical thumbnail.
- Final file will be resized to 1080x1350.

Text rules:
- Do not include any words, letters, captions, logos, fake text, typography, or readable marks.

Composition:
- Use the real uploaded speaker photo prominently.
- Make the subject larger and clearer for mobile.
- Strong vertical poster composition.
- Keep it clean, bold, and finished.
- No fake people.
`.trim();
}

async function generateGeminiImage({ prompt, references }) {
  const parts = [
    {
      text: prompt
    }
  ];

  references.forEach((ref) => {
    parts.push({
      inlineData: {
        mimeType: ref.mimeType,
        data: ref.base64
      }
    });
  });

  const response = await ai.models.generateContent({
    model: MODEL,
    contents: [
      {
        role: 'user',
        parts
      }
    ],
    config: {
      responseModalities: [Modality.IMAGE, Modality.TEXT]
    }
  });

  const candidate = response?.candidates?.[0];
  const responseParts = candidate?.content?.parts || [];

  const imagePart = responseParts.find((part) => {
    return part.inlineData && part.inlineData.data;
  });

  if (!imagePart) {
    const textPart = responseParts.find((part) => part.text);
    const text = textPart?.text || 'No image returned by Gemini.';
    throw new Error(text);
  }

  return {
    mimeType: imagePart.inlineData.mimeType || 'image/png',
    buffer: Buffer.from(imagePart.inlineData.data, 'base64')
  };
}

async function forceSize({ imageBuffer, width, height }) {
  return sharp(imageBuffer)
    .resize(width, height, {
      fit: 'cover',
      position: 'attention'
    })
    .png({
      quality: 95,
      compressionLevel: 8
    })
    .toBuffer();
}

function bufferToDataUrl(buffer, mimeType) {
  return `data:${mimeType};base64,${buffer.toString('base64')}`;
}

function fail(res, status, message) {
  return res.status(status).json({
    ok: false,
    message
  });
}

app.listen(PORT, () => {
  console.log(`Anchor Thumbnail AI backend running on port ${PORT}`);
});