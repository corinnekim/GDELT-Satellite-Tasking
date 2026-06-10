import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const dir = path.join(process.cwd(), 'public', 'data', 'dashboard');
    const files = fs.readdirSync(dir)
      .filter(f => f.match(/^daily_\d{8}\.json$/))
      .map(f => f.replace('daily_', '').replace('.json', ''))
      .sort();
    return NextResponse.json({ dates: files });
  } catch {
    return NextResponse.json({ dates: [] }, { status: 500 });
  }
}
