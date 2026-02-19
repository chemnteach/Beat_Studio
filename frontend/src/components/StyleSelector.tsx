/**
 * StyleSelector — grid of animation styles with recommendations highlighted.
 */
import { useState, useEffect } from 'react';
import axios from 'axios';
import type { AnimationStyle } from '../types';

interface Props {
  selected: string;
  onSelect: (style: string) => void;
  recommendedStyles?: string[];
}

const CATEGORY_ORDER = ['photorealistic', 'modern', 'traditional', 'stylized_realism', 'abstract'];

export function StyleSelector({ selected, onSelect, recommendedStyles = [] }: Props) {
  const [styles, setStyles] = useState<AnimationStyle[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeCategory, setActiveCategory] = useState<string>('all');

  useEffect(() => {
    void loadStyles();
  }, []);

  const loadStyles = async () => {
    setLoading(true);
    try {
      const { data } = await axios.get<{ styles: AnimationStyle[] }>('/api/video/styles');
      setStyles(data.styles);
    } catch { /* non-fatal */ } finally {
      setLoading(false);
    }
  };

  const categories = ['all', ...CATEGORY_ORDER.filter(c =>
    styles.some(s => s.category === c)
  )];

  const filtered = activeCategory === 'all'
    ? styles
    : styles.filter(s => s.category === activeCategory);

  return (
    <div className="style-selector" data-testid="style-selector">
      <h3>Animation Style</h3>

      {/* Category tabs */}
      <div className="category-tabs" data-testid="category-tabs">
        {categories.map(cat => (
          <button
            key={cat}
            className={activeCategory === cat ? 'active' : ''}
            onClick={() => setActiveCategory(cat)}
            data-testid={`category-${cat}`}
          >
            {cat === 'all' ? 'All' : cat.replace('_', ' ')}
          </button>
        ))}
      </div>

      {loading && <p>Loading styles…</p>}

      {/* Style grid */}
      <div className="style-grid" data-testid="style-grid">
        {filtered.map(style => (
          <div
            key={style.name}
            className={[
              'style-card',
              selected === style.name ? 'selected' : '',
              recommendedStyles.includes(style.name) ? 'recommended' : '',
            ].join(' ')}
            onClick={() => onSelect(style.name)}
            data-testid={`style-${style.name}`}
          >
            {recommendedStyles.includes(style.name) && (
              <span className="recommended-badge">Recommended</span>
            )}
            <h4>{style.display_name}</h4>
            <p className="best-for">{style.best_for}</p>
            <p className="backend-hint">{style.recommended_backend}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
